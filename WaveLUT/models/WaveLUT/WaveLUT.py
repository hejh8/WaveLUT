import torch
import torch.nn as nn

from .DN import denoise
# from .lut import LUT4DGenerator
from WaveLUT import WaveLUT_transform

from models.dwt import DWT, IWT, get_Fre
# from .mod import HFRM

import torch.nn.functional as F

class LUT4DGenerator(nn.Module):
    r"""The 4DLUT generator module.

    Args:
        n_channels (int): Number of input channels.
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_channels, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_channels), bias=False)

        self.n_channels = n_channels
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
        
        self.drop = nn.Dropout(p=0.5)

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) and develop it into 4D version.

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_channels)]),
                dim=0).div(self.n_vertices - 1).flip(0)[:-1],
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_channels)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x,x_l):
        weights_x = self.weights_generator(x)
        weights_l = self.weights_generator(x_l)
        
        weights = 0.1*weights_l+0.9*weights_x
        # weights = 0.2*weights_l+0.8*weights_x

        # shortcut = weights_x
        # # x = self.norm1(x)
        # q_x = weights_l 
        # # n1,n2= weights_x 
        # k_x = weights_x
        # attn_x = (q_x @ k_x.transpose(-1, -2)).softmax(dim=-1)
        # x_ = attn_x @ k_x
        # weights = shortcut + self.drop(x_)

        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_channels))
        return weights, luts

class LightBackbone3D(nn.Module):
    def __init__(self, input_resolution, train_resolution, n_base_feats, extra_pooling=False, **kwargs):
        super(LightBackbone3D, self).__init__()
        self.n_feats = n_base_feats
        self.extra_pooling = extra_pooling

        # Conv_net
        self.conv = nn.Sequential(nn.Conv3d(3, self.n_feats, kernel_size=3, stride=(1, 2, 2), padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats, affine=True),
                                  nn.Conv3d(self.n_feats, self.n_feats * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats * 2, affine=True),
                                  nn.Conv3d(self.n_feats * 2, self.n_feats * 4, kernel_size=3, stride=(1, 2, 2),
                                            padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats * 4, affine=True),
                                  nn.Conv3d(self.n_feats * 4, self.n_feats * 8, kernel_size=3, stride=(1, 2, 2),
                                            padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats * 8, affine=True),
                                  nn.Conv3d(self.n_feats * 8, self.n_feats * 8, kernel_size=3, stride=(1, 2, 2),
                                            padding=1),
                                  nn.LeakyReLU(0.2))

        # self.linear_q = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        self.upsample1 = nn.Upsample(size=(512, 960), mode='bilinear', align_corners=False)
        
        # dropout and pooling for LUT geneator
        self.drop = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.out_channels = self.n_feats * 8 * (16 if extra_pooling else 7 * (input_resolution[0] // 32) * (input_resolution[1] // 32))
        self.pool_intensity = nn.AdaptiveAvgPool3d((1, input_resolution[1], input_resolution[0]))
        self.pool_intensity_train = nn.AdaptiveAvgPool3d((1, train_resolution[1], train_resolution[0]))

        # Deconv_net
        self.deconv = nn.Sequential(nn.InstanceNorm3d(self.n_feats * 8, affine=True),
            nn.ConvTranspose3d(self.n_feats * 8, self.n_feats * 8, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 8, affine=True),
            nn.ConvTranspose3d(self.n_feats * 8, self.n_feats * 4, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 4, affine=True),
            nn.ConvTranspose3d(self.n_feats * 4, self.n_feats * 2, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 2, affine=True),
            nn.ConvTranspose3d(self.n_feats * 2, self.n_feats, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats, affine=True),
            nn.ConvTranspose3d(self.n_feats, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))

    def forward(self,LL, videos, if_train):
        b, c, t, h, w = videos.shape

        # Extract feature
        x = self.conv(videos)
        x_L = self.conv(LL)
        if self.extra_pooling:
            codes = self.drop(x)
            codes = self.pool(codes).view(b, -1)
            codes_l = self.drop(x_L)
            codes_l = self.pool(codes_l).view(b, -1)
        else:
            codes = self.drop(x).view(b, -1)
            codes_l = self.drop(x_L).view(b, -1)

        # Create intensity map
        # intensity_map = self.deconv(x)
        intensity_map_0 = self.deconv(x)
        intensity_map_1 = self.deconv(x_L)
        # intensity_map_=0.1*intensity_map_1+0.9*intensity_map_0
        
        shortcut = intensity_map_0
        q_x = intensity_map_1
        b1, c1, t1, h1, w1= q_x.shape
        q_x = q_x.view(b1 * t1, c1, h1, w1)    
             
        # shortcut = intensity_map_0
        # # x = self.norm1(x)
        # q_x = intensity_map_1
        # b1, c1, t1, h1, w1= q_x.shape
        # q_x = q_x.view(b1 * t1, c1, h1, w1)  
        # q_x = self.upsample(q_x)  
        # q_x = q_x.view(b1, c1, t1, h, w) 
        # # print(q_x.shape)
        # k_x = intensity_map_0
        # attn_x = (q_x @ k_x.transpose(-1, -2)).softmax(dim=-1)
        # x_ = attn_x @ k_x
        # # x = x.squeeze(dim=2)
        # x_ = shortcut + self.drop(x_)
        # intensity_map = x_[:, :, 1:t, :, :]

        # intensity_map = intensity_map[:, :, 1:t, :, :]
        
        if if_train:
            q_x = self.upsample(q_x)  # 对空间维度进行上采样
            q_x = q_x.view(b1, c1, t1, 256, 256) 
            k_x = intensity_map_0
            attn_x = (q_x @ k_x.transpose(-1, -2)).softmax(dim=-1)
            x_ = attn_x @ k_x
            x_ = shortcut + self.drop(x_)
            intensity_map = x_[:, :, 1:t, :, :]
            
            intensity_map = self.pool_intensity_train(intensity_map)
        else:
            q_x = self.upsample1(q_x)  # 对空间维度进行上采样
            q_x = q_x.view(b1, c1, t1, 512, 960) 
            k_x = intensity_map_0
            attn_x = (q_x @ k_x.transpose(-1, -2)).softmax(dim=-1)
            x_ = attn_x @ k_x
            x_ = shortcut + self.drop(x_)
            intensity_map = x_[:, :, 1:t, :, :]
            # intensity_map = self.pool_intensity_train(intensity_map)
            intensity_map = self.pool_intensity(intensity_map)
        # intensity_map = self.intensity_norm(b, intensity_map)
         
        intensity_map_list = []
        
        for i in range(t):
            intensity_map_list.append(intensity_map)
        intensity_map = torch.cat(intensity_map_list, dim=2)

        return codes,codes_l, intensity_map



class WaveLUT_LLVE(nn.Module):

    def __init__(self, 
        input_resolution,
        train_resolution,
        n_ranks = 3,
        n_vertices_4d = 33,
        n_base_feats = 8,
        smooth_factor = 0.,
        monotonicity_factor = 10.0):
        super(WaveLUT_LLVE, self).__init__()
        self.n_ranks = n_ranks
        self.n_vertices_4d = n_vertices_4d
        self.n_base_feats = n_base_feats
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        
        # self.backbone_name = backbone_name 
        self.dwt = DWT()
        self.iwt = IWT()
        self.fft = get_Fre()
        self.backbone = LightBackbone3D(input_resolution=input_resolution, train_resolution=train_resolution,
                                        extra_pooling=True, n_base_feats=n_base_feats)
        self.denoise = denoise()
        self.LUTGenerator = LUT4DGenerator(4, 3, n_vertices_4d, self.backbone.out_channels, n_ranks)
        self.init_weights()

    def init_weights(self):
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        # if self.backbone_name not in ['res18']:
        #     self.apply(special_initilization)
        self.LUTGenerator.init_weights()
    
    def forward(self, videos, if_train=True):
        min_max = (0, 1)
        n,c,t,h,w=videos.shape
        self.L_dwt = self.dwt(videos)
        self.LL, self.LH = self.L_dwt[:n, ...],self.L_dwt[n:, ...]

        codes,codes_l, intensity_map = self.backbone(self.LL,videos, if_train)
        weights, luts = self.LUTGenerator(codes,codes_l)
        
        context_videos = torch.cat((videos, intensity_map), dim=1)
        outputs = WaveLUT_transform(context_videos, luts)
        outputs_clamp = torch.clamp(outputs, min_max[0], min_max[1])
        clear_outputs = self.denoise(outputs_clamp)

        return clear_outputs