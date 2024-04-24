import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1

class MPD(nn.Module):
    def __init__(self, p):
        super(MPD, self).__init__()
        self.p = p
        dims = [1, 32, 128, 512, 1024]
        
        self.stage1 = nn.ModuleList()
        for i in range(4):
            layer = nn.Sequential(
                weight_norm(nn.Conv2d(dims[i], dims[i+1], kernel_size=(5,1), stride=(3,1), padding=(2,0))),
                nn.LeakyReLU(LRELU_SLOPE),
            )
            self.stage1.append(layer)
        self.stage2 = nn.Sequential(
            weight_norm(nn.Conv2d(1024,1024, kernel_size=(5,1), stride=(1,1), padding=(2,0))),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        self.out =  nn.Sequential(
            weight_norm(nn.Conv2d(1024, 1, kernel_size=(3,1), stride=(1,1), padding=(1,0))),
        )
           
    def forward(self, W):
        fmap = []
        # -> (B,T')
        b, t = W.shape
        if t % self.p != 0: # pad first
            n_pad = self.p - (t % self.p)
            W = F.pad(W, (0, n_pad), "reflect")
            t = t + n_pad
        W = W.view(b, 1, t // self.p, self.p)
        # -> (B,1,T/p,p) 
        for layer in self.stage1:
            W = layer(W)
            fmap.append(W)
        W = self.stage2(W)
        fmap.append(W)
        W = self.out(W)
        fmap.append(W)
        W = torch.flatten(W, 1, -1)
        # -> (B, many)
        return W, fmap
    
class MRD(nn.Module):
    def __init__(self, nfft, hopsize, format):
        # format: 'mag' or 'pahse'
        super(MRD, self).__init__()
        self.nfft = nfft
        self.hopsize = hopsize
        self.format = format
        self.stage1 = nn.Sequential(
            weight_norm(nn.Conv2d(1, 64, kernel_size=(7,5), stride=(2,2), padding=(3,2))),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        klist = [(5,3),(5,3),(3,3),(3,3)]
        slist = [(2,1),(2,2),(2,1),(2,2)]
        plist = [(2,1),(2,1),(1,1),(1,1)]
        self.stage2 = nn.ModuleList()
        for i in range(4):
            layer = nn.Sequential(
                weight_norm(nn.Conv2d(64, 64, kernel_size=klist[i], stride=slist[i], padding=plist[i])),
                nn.LeakyReLU(LRELU_SLOPE),
            )
            self.stage2.append(layer)
        self.out =  nn.Sequential(
            weight_norm(nn.Conv2d(64, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1))),
        )
    def forward(self, W):
        fmap = []
        # -> (B,T')
        W = torch.stft(
            W, n_fft=self.nfft, 
            hop_length=self.hopsize, 
            window=torch.ones(self.nfft).to(W.device),
            onesided=True, 
            return_complex=True
        )
        # -> (B,F,T)
        if self.format == 'mag':
            W = torch.abs(W)
        else:
            W = torch.angle(W)
        W = W.unsqueeze(1)
        # -> (B,1,F,T)
        W = self.stage1(W)
        fmap.append(W)
        for layer in self.stage2:
            W = layer(W)
            fmap.append(W)
        W = self.out(W)
        fmap.append(W)
        W = torch.flatten(W, 1, -1)
        return W, fmap


class APdisc(nn.Module):
    def __init__(self):
        super(APdisc, self).__init__()
        p_list = [2, 3, 5, 7, 11]
        self.MPD_list = nn.ModuleList([MPD(p) for p in p_list])
        self.nfftlist = [512,1024,2048]
        self.nhoplist = [128,256,512]
        # self.nwinlist = [512,1024,2048]
        self.MRAD_list = nn.ModuleList([MRD(nfft, hopsize, 'mag') for nfft, hopsize in zip(self.nfftlist, self.nhoplist)])
        self.MRPD_list = nn.ModuleList([MRD(nfft, hopsize, 'phase') for nfft, hopsize in zip(self.nfftlist, self.nhoplist)])

    def forward(self, W):
        # -> (B,T)
        out_list = []
        fmap_list = []
        for MPD in self.MPD_list:
            MPD_out, MPD_fmap= MPD(W)
            out_list.append(MPD_out) 
            fmap_list.append(MPD_fmap)
        for MRAD, MRPD in zip(self.MRAD_list, self.MRPD_list):
            MRAD_out, MRAD_fmap = MRAD(W)
            MRPD_out, MRPD_fmap = MRPD(W)
            out_list.append(MRAD_out)
            out_list.append(MRPD_out)
            fmap_list.append(MRAD_fmap)
            fmap_list.append(MRPD_fmap)
        return out_list, fmap_list
