import os
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
import torch.nn.functional as F
from models.generator import APNet
from models.discriminator import APdisc
from utils.func import fAW, feature_loss, discriminator_loss, generator_loss
from torch.utils.tensorboard import SummaryWriter
import datetime
import random

from dataloader import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_avail", type=str, default="0", help="available GPUs")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=8000, help="cut length, default is 8000")
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--steps", type=int, default=500000, help="number of steps where training stops")
parser.add_argument("--data_dir", type=str, default='/data/hdd0/xinan.chen/VCTK_wav_single_trim',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./ckpts',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[45, 100, 45],
                    help="weights of Amplitude Spectrum Loss, Phase Spectrum Loss, and Complex Spectrum Loss")

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, train_ds, test_ds, gpu_id: int):
        self.n_fft = 1024
        self.win = 320
        self.hop = 80
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model = APNet(Freqbin=513).cuda()
        self.discriminator = APdisc().cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr, betas=(0.8, 0.99), weight_decay=0.01)
        self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=args.init_lr, betas=(0.8, 0.99), weight_decay=0.01)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id

    def forward_generator_step(self, clean, noisy):
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        #  [T, batchsize] * [batchsize,] -> [T, batchsize]
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )
        # 输入 (batchsize,t)
        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop, # 窗口的跳跃步长，也就是相邻窗口之间的样本数
            win_length=self.win,
            window=torch.hann_window(self.win).to(self.gpu_id),
            onesided=True, # onesided=True 表示只计算并返回正频率。
            return_complex=True,
        )

        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            win_length=self.win,
            window=torch.hann_window(self.win).to(self.gpu_id),
            onesided=True,
            return_complex=True,
        )

        # -> (batchsize,F,T)
        clean_real = clean_spec.real
        clean_imag = clean_spec.imag
        clean_mag = torch.log(torch.abs(clean_spec).clamp(1e-8))
        clean_pha = torch.angle(clean_spec)
        # -> (batchsize,F,T)

        noisy_mag = torch.log(torch.abs(noisy_spec).clamp(1e-8))
        noisy_pha = torch.angle(noisy_spec)
        # -> (batchsize,F,T)

        est_mag, est_pha = self.model(noisy_mag, noisy_pha)
        # -> (batchsize,F,T)

        # 还原
        est_real = torch.exp(est_mag) * torch.cos(est_pha)
        est_imag = torch.exp(est_mag) * torch.sin(est_pha)
        # -> (batchsize,F,T)
        est_spec = torch.stack([est_real, est_imag], dim=3)
        # -> (batchsize,F,T,2)
        est_spec=torch.view_as_complex(est_spec)
        est_audio = torch.istft(
            est_spec,
            self.n_fft,
            self.hop,
            win_length=self.win,
            window=torch.hann_window(self.win).to(self.gpu_id),
            onesided=True,
        )
        assert est_audio.size(-1) == clean.size(-1)
        # -> (batchsize,t)

        # return 除audio都为(batchsize,F,T)
        return {
            "est_pha": est_pha,
            "est_mag": est_mag,
            "est_audio": est_audio,
            "clean_pha": clean_pha,
            "clean_mag": clean_mag,
            "clean_audio": clean,
            "clean_complex": clean_spec,
            "est_complex": est_spec,
        }
    
    
    def calculate_generator_loss(self, generator_outputs):

        predict_fake_metric, fmap_est = self.discriminator(
            generator_outputs["est_audio"]
        )
        loss_adv = generator_loss(predict_fake_metric)

        _, fmap_clean = self.discriminator(generator_outputs["clean_audio"])
        loss_FM = feature_loss(fmap_est, fmap_clean)

        loss_A = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        ) 

        IP = generator_outputs["est_pha"]-generator_outputs["clean_pha"]
        GD = torch.diff(generator_outputs["est_pha"], dim = 1) - torch.diff(generator_outputs["clean_pha"], dim = 1)
        IAF = torch.diff(generator_outputs["est_pha"], dim = 2) - torch.diff(generator_outputs["clean_pha"], dim = 2)
        loss_P= torch.mean(fAW(IP)) + torch.mean(fAW(GD)) + torch.mean(fAW(IAF))

    
        loss_C = F.mse_loss(
            generator_outputs["est_complex"].real, generator_outputs["clean_complex"].real
        ) + F.mse_loss(
            generator_outputs["est_complex"].imag, generator_outputs["clean_complex"].imag
        )

        # time_loss = torch.mean(
        #     torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        # )

        loss = (
            args.loss_weights[0] * loss_A
            + args.loss_weights[1] * loss_P
            + args.loss_weights[2] * loss_C
            + loss_adv
            + loss_FM
        )

        return loss
    
    def calculate_discriminator_loss(self, generator_outputs):
        predict_fake_metric_est, _ = self.discriminator(generator_outputs["est_audio"].detach())
        predict_fake_metric_clean, _ = self.discriminator(generator_outputs["clean_audio"])
    
        return discriminator_loss(predict_fake_metric_clean, predict_fake_metric_est)


    def train_step(self, batch):
        # Train generator
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        # one_labels = torch.ones(args.batch_size, 11).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        # generator_outputs["one_labels"] = one_labels
        # # 检测generator_outputs
        # for key in generator_outputs.keys():
        #     assert generator_outputs[key].size(0)==args.batch_size
        # print("done")
            
        loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        self.optimizer_disc.zero_grad()
        discrim_loss_metric.backward()
        self.optimizer_disc.step()
        assert not torch.isnan(loss).any()
        assert not torch.isnan(discrim_loss_metric).any()

        return loss.item(), discrim_loss_metric.item()
    
    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        # one_labels = torch.ones(clean.size(0), 11).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        # generator_outputs["one_labels"] = one_labels
        # generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        assert not torch.isnan(loss).any()
        assert not torch.isnan(discrim_loss_metric).any()

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        # template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        # logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg, disc_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.999
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=1, gamma=0.999
        )
        best_gen_loss = float(1000000)
        now_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
        steps = 0
        epoch = 0
        if self.gpu_id == 0:
            writer = SummaryWriter(log_dir=os.path.join("runs", now_time))    

        while steps < args.steps and epoch < args.epochs:
            epoch += 1
            if self.gpu_id == 0:
                print("Epoch start:", epoch, now_time, "steps:", steps)
            self.model.train()
            self.discriminator.train()
            gen_loss_total = 0.0
            disc_loss_total = 0.0
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                steps += 1
                loss, disc_loss = self.train_step(batch)
                gen_loss_total += loss
                disc_loss_total += disc_loss
                if steps >= args.steps:
                    break
            gen_loss_train = gen_loss_total / step
            disc_loss_train= disc_loss_total / step
            gen_loss_test, disc_loss_test= self.test() # 只用生成器loss

            if self.gpu_id == 0:
                writer.add_scalar('gen_loss_train',gen_loss_train , epoch) 
                writer.add_scalar('disc_loss_train',disc_loss_train , epoch)
                writer.add_scalar('gen_loss_test',gen_loss_test , epoch)
                writer.add_scalar('disc_loss_test',disc_loss_test , epoch)
                if gen_loss_test < best_gen_loss:
                    # torch.save(self.model.module.state_dict(), path)
                    best_model = self.model.module.state_dict()
                    best_gen_loss = gen_loss_test
                    best_epoch = epoch
                    writer.add_scalar('best_loss', gen_loss_test, epoch)
                    # best_path = path
                print("Epoch end, genloss:", gen_loss_train, gen_loss_test)
            # 更新学习率
            scheduler_G.step() 
            scheduler_D.step()
            
        # 将最好的模型复制到best_ckpt文件夹下
        if self.gpu_id == 0:
            print("Best epoch: {}, Best loss: {}".format(best_epoch, best_gen_loss))
            save_path = os.path.join(args.save_model_dir, "best_ckpt_"+now_time)
            torch.save(best_model, save_path)
            writer.close()

def ddp_setup(rank, world_size):
    # 设置主节点的地址和端口号
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    # 初始化进程组 NCCL通信
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, args):
    set_seed(3407)
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)
        
    train_ds, test_ds = load_data(
        args.data_dir, args.batch_size, n_cpu=2, cut_len=args.cut_len
    )
    trainer = Trainer(train_ds, test_ds, rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_avail
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    mp.spawn(main, args=(world_size, args), nprocs=world_size)