import torch
import torch.nn as nn
import torch.nn.functional as F

class InforComu(nn.Module):
        # 1d version
        def __init__(self, src_channel, tgt_channel):            
            super(InforComu, self).__init__()
            self.comu_conv = nn.Conv1d(src_channel, tgt_channel, kernel_size=1)
        
        def forward(self, src, tgt):
            outputs=tgt*torch.tanh(self.comu_conv(src))

            return outputs

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first": 
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class Block(nn.Module):
    """ ConvNeXt Block 1d
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 3 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(3 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
        #                             requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x # (N, F, T)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        x = input + x
        return x
    
class APNet(nn.Module):
    def __init__(self, Freqbin=513):
        super(APNet, self).__init__()
        self.B = Freqbin
        self.Ain = nn.Sequential(
            nn.Conv1d(Freqbin, self.B, kernel_size=3, padding=1),
            LayerNorm(self.B, eps=1e-6, data_format="channels_first")
        )
        self.Pin = nn.Sequential(
            nn.Conv1d(Freqbin, self.B, kernel_size=3, padding=1),
            LayerNorm(self.B, eps=1e-6, data_format="channels_first")
        )
        self.Ablocks = nn.ModuleList([Block(self.B) for _ in range(8)])
        self.Pblocks = nn.ModuleList([Block(self.B) for _ in range(8)])
        self.Aout = nn.Sequential(
            LayerNorm(self.B, eps=1e-6, data_format="channels_first"),
            nn.Conv1d(self.B, Freqbin, kernel_size=3, padding=1)
        )
        self.Pout = LayerNorm(self.B, eps=1e-6, data_format="channels_first")
        self.PoutR = nn.Conv1d(self.B, Freqbin, kernel_size=3, padding=1)
        self.PoutI = nn.Conv1d(self.B, Freqbin, kernel_size=3, padding=1)

        # self.p2a_comu = InforComu(self.B, self.B)
        # self.a2p_comu = InforComu(self.B, self.B)

    def forward(self, A, P):
        A_in = A
        A = self.Ain(A)
        P = self.Pin(P)
        for idx in range(len(self.Ablocks)):
            # Ac = self.p2a_comu(P, A)
            # Pc = self.a2p_comu(A, P)
            Ac = A + P
            Pc = P + A
            A = self.Ablocks[idx](Ac)
            P = self.Pblocks[idx](Pc)
        A = self.Aout(A)+A_in
        P = self.Pout(P)
        P_R = self.PoutR(P)
        P_I = self.PoutI(P)
        P = torch.atan2(P_I, P_R)
        return A, P
