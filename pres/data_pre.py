# 数据预处理，将flac格式的音频文件转换为wav格式，48khz降采样至16kHz，并产生8/4/2khz低通滤波信号作为输入，分为训练集和测试集
import os
import argparse
import torchaudio
import torch
import numpy as np
from torchaudio.transforms import Resample, Vad
from tqdm import tqdm
# from scipy.signal import iirfilter, filtfilt
from natsort import natsorted


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="multi", help='single or multi speakers')
parser.add_argument('--VCTK_data_path', type=str, default='', help='data path')
parser.add_argument('--save_path', type=str, default='', help='save path')
# parser.add_argument('--downsample_rate', type=int, default=2, help='downsample rate')

# for single file numbers, for multi speaker numbers
parser.add_argument('--train_num', type=int, default=100, help='train num')
parser.add_argument('--test_num', type=int, default=8, help='test num')

args = parser.parse_args()


if __name__ == '__main__':
    # torchaudio.set_audio_backend("sox_io") 

    input_sample_rate = 48000
    output_sample_rate = 16000

    resample = Resample(input_sample_rate, output_sample_rate)
    
    if args.mode == "single":
        data_path = os.path.join(args.VCTK_data_path,"p225")

        # 检查文件数量是否正确
        count = 0
        for file in os.listdir(data_path):
            if file.endswith("_mic1.flac"):
                count += 1
        print("total file num:", count)
        assert count == args.train_num + args.test_num
    else:
        data_path = args.VCTK_data_path

        # 检查文件夹数量是否正确
        count = 0
        for folder in os.listdir(data_path):
            count += 1
        print("total folder num:", count)
        assert count == args.train_num + args.test_num
    save_train_clean_path = os.path.join(args.save_path,"train","clean")
    # save_train_noisy_path = os.path.join(args.save_path,"train","noisy")
    save_test_clean_path = os.path.join(args.save_path,"test","clean")
    # save_test_noisy_path = os.path.join(args.save_path,"test","noisy")

    os.makedirs(save_train_clean_path, exist_ok=True)
    # os.makedirs(save_train_noisy_path, exist_ok=True)
    os.makedirs(save_test_clean_path, exist_ok=True)
    # os.makedirs(save_test_noisy_path, exist_ok=True)

    
    # flac to wav
    idx = -1 if args.mode == "multi" else 0
    for root, dirs, files in os.walk(data_path):
        if args.mode == "multi":
            dirs[:] = natsorted(dirs)
            idx += 1
        files = natsorted(files)
        for file in tqdm(files):
            if file.endswith("_mic1.flac"):
                if args.mode == "single" :
                    idx += 1
                # if idx > args.train_num:
                #     file_path = os.path.join(root, file)
                #     wav, sr = torchaudio.load(file_path)
                #     wav = resample(wav)
                #     torchaudio.save(os.path.join(save_test_clean_path, file[:-5] + ".wav"), wav, sample_rate=output_sample_rate)

                file_path = os.path.join(root, file)
                wav, sr = torchaudio.load(file_path)
                # [c,t]
                wav = resample(wav)

                # trim
                transform = Vad(sample_rate=output_sample_rate)
                waveform_start_trim = transform(wav)
                flipped_audio = torch.flip(waveform_start_trim , [1])
                waveform_end_trim = transform(flipped_audio)
                wav = torch.flip(waveform_end_trim, [1])

                # save clean wav
                if idx <= args.train_num:
                    torchaudio.save(os.path.join(save_train_clean_path, file[:-5] + ".wav"), wav, sample_rate=output_sample_rate)
                else:
                    torchaudio.save(os.path.join(save_test_clean_path, file[:-5] + ".wav"), wav, sample_rate=output_sample_rate)

                # save noisy wav (先滤波)
                order = 8
                ripple = 0.05
                hi = 1/args.downsample_rate
                b, a =iirfilter(order, hi, rp=ripple, btype='lowpass',ftype='cheby1', output='ba')
                wav_l = filtfilt(b, a, wav.numpy())
                wav_l = torch.from_numpy(wav_l.copy()).to(torch.float32)

                assert len(wav_l) == len(wav)

                if idx <= args.train_num:
                    torchaudio.save(os.path.join(save_train_noisy_path, file[:-5] + ".wav"), wav_l, sample_rate=output_sample_rate)
                else:
                    torchaudio.save(os.path.join(save_test_noisy_path, file[:-5] + ".wav"), wav_l, sample_rate=output_sample_rate)
        
    print("done")
