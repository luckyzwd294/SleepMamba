import torch.nn as nn
import torch
from layers.Embedding import PositionalEmbedding
from mamba_ssm import Mamba
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio , 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes //ratio , in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        res = self.sigmoid(out)
        return res


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        res = self.sigmoid(x)
        return res


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    

# 通过两个大小核提取信息
class MRCNN2(nn.Module):
    def __init__(self, in_channels=2, out_channels=128, drate=0.5):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=1, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
        )
        self.features2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 64
        self.cbam1 = CBAM(in_planes=self.inplanes, ratio=16, kernel_size=7)
        self.cbam2 = CBAM(in_planes=self.inplanes, ratio=16, kernel_size=7)


    def forward(self, x):
        # print('before MRCNN', x.shape)
        x1 = self.features1(x)
        x2 = self.features2(x)
        x1 = self.cbam1(x1)
        x2 = self.cbam2(x2)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        return x_concat



class MambaEncoder(nn.Module):
    def __init__(self, EncoderLayer, norm_layer=None):
        super().__init__()
        self.encoder = nn.ModuleList(EncoderLayer)
        self.norm = norm_layer

    def forward(self, x):
        for mamba in self.encoder:
            x = mamba(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer2(nn.Module):
    def __init__(self, d_model, in_features=128, out_features=128, d_state=16, d_conv=2, d_ff=None, drate=0.5, activation=F.relu):
        super().__init__()
        # Mamba本身是包含Gate的ssm，通过activation(linear layer)实现的门控增加非线性
        self.man = Mamba(
            d_model=d_model, # Model dimension
            d_state=d_state, # SSM state expansion factor
            d_conv=d_conv,
            expand=1,
        )
        self.man2 = Mamba(
            d_model=64,  # Model dimension
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,
            expand=1,
        )

        self.activation = F.relu if activation == "relu" else F.gelu

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(d_model)

        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        #
        # self.gate = Gate_Layer2(d_model=d_model, in_features=in_features, out_features=out_features, hidden=d_ff, drate=drate)
        # self.gate_flip = Gate_Layer2(d_model=d_model, in_features=in_features, out_features=out_features, hidden=d_ff, drate=drate)
        #
        self.dropout = nn.Dropout(drate)


    def forward(self, x):

        input = x.permute(0, 2, 1)
        mamba = self.man(x)
        mamba = self.norm1(mamba)

        # inter-modal
        mamba2 = self.man2(input)
        mamba2 = self.norm2(mamba2)

        mamba2 = mamba2.permute(0, 2, 1)
        x = mamba + mamba2

        # y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # y = self.norm3(x+y)

        return x



class EpochMamba2(nn.Module):
    def __init__(self, configs):
        super(EpochMamba2, self).__init__()
        self.mrcnn = MRCNN2()
        self.pos_emb = PositionalEmbedding(d_model=256)
        d_model = configs.d_model
        d_ff = configs.d_ff or 4 * d_model

        self.dense = nn.Linear(in_features=configs.dense, out_features=256, bias=True)

        self.encoder = MambaEncoder(
            [
                EncoderLayer2(
                    d_model=d_model,
                    in_features=configs.in_features,
                    out_features=configs.out_features,
                    d_state=configs.d_state,
                    d_conv=3,
                    d_ff=d_ff,
                    drate=configs.drate
                ) for _ in range(configs.num_encoder)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.conv1 = nn.Conv1d(10, 128, kernel_size=1)


    def forward(self, x):
        x = self.mrcnn(x)
        x = self.dense(x)  # [B, 128, 142] -> [B, 128, 256]
        x = self.encoder(x)
        return x


class SequenceLayer2(nn.Module):
    def __init__(self, configs, d_model, d_state, d_ff=None, drate=0.5):
        super().__init__()
        # Sequence之间的mamba，考虑不做CNN

        self.man = Mamba(
            d_model=d_model, # Model dimension
            d_state=d_state, # SSM state expansion factor
            d_conv=2,
            expand=1,
        )
        self.rman = Mamba(
            d_model=d_model,  # Model dimension
            d_state=d_state,  # SSM state expansion factor
            d_conv=2,
            expand=1,
        )

        # self.gate = Gate_Layer(d_model=d_model, in_features=256, out_features=256, hidden=1024, drate=drate)
        # self.gate_flip = Gate_Layer(d_model=d_model, in_features=256, out_features=256, hidden=1024, drate=drate)

    def forward(self, x):
        mamba = self.man(x)

        x_flip = x.flip(dims=[1])
        mamba_flip = self.rman(x_flip)
        mamba_flip = mamba_flip.flip(dims=[1])
        x = mamba + mamba_flip

        return x


class SequenceMamba2(nn.Module):
    def __init__(self, configs):
        super().__init__()
        d_model = configs.s_model
        d_ff = configs.d_ff or 4 * d_model

        self.epochmamba = nn.ModuleList([EpochMamba2(configs) for _ in range(configs.window)])

        d_state = 32

        self.multiencoder = SequenceLayer2(configs, d_model=d_model, d_state=d_state, d_ff=d_ff)

        self.flatten = nn.Flatten(start_dim=-2)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(128*128, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5)
        # )
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        B, E, C, L = x.shape # [B, Epochs_num, Channel, L] 初始应该是[B, 5, 3, 3000]
        inputs = []
        for i in range(0, E):
            sub = x[:, i, :, :]             # [B, C, L]
            sub = self.epochmamba[i](sub)      # [B, C, L] -> [B, C', d]
            sub = sub.mean(dim=1)
            # sub = self.flatten(sub)         # [B, C' * d]
            # sub = self.fc1(sub)             # [B, C' * d] -> [B, 1024]
            inputs.append(sub)
        fusion_output = torch.stack(inputs, dim=1)
        fusion_output = self.multiencoder(fusion_output)

        fusion_output = self.fc2(fusion_output)
        return fusion_output