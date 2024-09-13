import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class dfus_block(nn.Module):
    def __init__(self):
        super(dfus_block, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(inplace=True))

        self.convc3 = nn.Sequential(nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.convd3 = nn.Sequential(nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                    nn.ReLU(inplace=True))
        self.convc3d3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                      nn.ReLU(inplace=True))
        self.convd3c3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        fea1 = self.conv1(x)

        feac3 = self.convc3(fea1)
        fead3 = self.convd3(fea1)
        feac3d3 = self.convc3d3(feac3)
        fead3c3 = self.convd3c3(fead3)

        fea = torch.cat([feac3, fead3, feac3d3, fead3c3], dim=1)
        fea = self.conv2(fea)

        return torch.cat([fea1, fea], dim=1)


class denoise(nn.Module):
    def __init__(self):
        super(denoise, self).__init__()
        # ddfn Feature _extraction
        self.convc3 = nn.Sequential(nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.convd3 = nn.Sequential(nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                    nn.ReLU(inplace=True))
        self.convc3d3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                      nn.ReLU(inplace=True))
        self.convd3c3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        # ddfn Feature_integration
        dfus_block_generator = functools.partial(dfus_block)
        self.dfus = make_layer(dfus_block_generator, 1)

        # ddfn Reconstruction
        self.Reconstruction = nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feac3 = self.convc3(x)
        fead3 = self.convd3(x)
        feac3d3 = self.convc3d3(feac3)
        fead3c3 = self.convd3c3(fead3)
        fea = torch.cat([feac3, fead3, feac3d3, fead3c3], dim=1)

        fea = self.dfus(fea)
        fea = self.Reconstruction(fea)

        return fea


