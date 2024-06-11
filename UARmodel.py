import torch
import torch.nn as nn
from resnet import resnet50
from einops import repeat
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SubSpace(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super(SubSpace, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class ULSAM(nn.Module):
    def __init__(self, nin: int, num_splits: int, reduction: int) -> None:
        super(ULSAM, self).__init__()
        assert nin % num_splits == 0
        self.nin = nin
        self.num_splits = num_splits
        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits), reduction) for _ in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sub_feat = torch.chunk(x, self.num_splits, dim=1)
        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))
        out = torch.cat(out, dim=1)

        return out


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class UARmodel(nn.Module):
    def __init__(self):
        super(UARmodel, self).__init__()
        self.resnet_features = resnet50(pretrained=True)
        self.conv5 = nn.Conv2d(2048, 1024, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.rl5 = nn.ReLU()
        self.down5 = nn.Conv2d(1024, 128, 1, bias=False)
        self.ulsam5 = ULSAM(1024, 32, 8)
        self.conv4 = nn.Conv2d(1024, 512, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.rl4 = nn.ReLU()
        self.down4 = nn.Conv2d(512, 128, 1, bias=False)
        self.ulsam4 = ULSAM(512, 16, 8)
        self.conv3 = nn.Conv2d(512, 256, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.rl3 = nn.ReLU()
        self.down3 = nn.Conv2d(256, 128, 1, bias=False)
        self.ulsam3 = ULSAM(256, 8, 8)
        self.conv2 = nn.Conv2d(256, 128, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.rl2 = nn.ReLU()
        self.down2 = nn.Conv2d(128, 128, 1, bias=False)
        self.ulsam2 = ULSAM(128, 4, 8)

        self.conv_weight = nn.Conv2d(128, 3, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.S4 = nn.Conv2d(128, 128, 1, bias=False)
        self.S3 = nn.Conv2d(128, 128, 1, bias=False)
        self.S2 = nn.Conv2d(128, 128, 1, bias=False)

        self.transformer = Block(dim=128, num_heads=8, mlp_ratio=4, qkv_bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, 166, 128))
        self.scale_p4_embedding = torch.full((1, 9, 128), 0.0001)
        self.scale_p3_embedding = torch.full((1, 36, 128), 0.0002)
        self.scale_p2_embedding = torch.full((1, 121, 128), 0.0003)

        self.avg_pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.weight4 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid4 = nn.Sigmoid()
        self.avg_pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.weight3 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.weight2 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        output, e2, e3, e4, e5 = self.resnet_features(x)

        e2 = self.conv2(e2)
        e2 = self.bn2(e2)
        e2 = self.rl2(e2)

        e3 = self.conv3(e3)
        e3 = self.bn3(e3)
        e3 = self.rl3(e3)

        e4 = self.conv4(e4)
        e4 = self.bn4(e4)
        e4 = self.rl4(e4)

        e5 = self.conv5(e5)
        e5 = self.bn5(e5)
        e5 = self.rl5(e5)

        p5 = self.down5(self.ulsam5(e5))
        p4 = F.interpolate(p5, size=e4.shape[-2:], mode='nearest') + self.down4(self.ulsam4(e4))
        p3 = F.interpolate(p4, size=e3.shape[-2:], mode='nearest') + self.down3(self.ulsam3(e3))
        p2 = F.interpolate(p3, size=e2.shape[-2:], mode='nearest') + self.down2(self.ulsam2(e2))

        out_size = p3.shape[-2:]
        S_p4 = F.interpolate(p4, size=out_size, mode='nearest')
        S_p3 = F.adaptive_max_pool2d(p3, output_size=out_size)
        S_p2 = F.adaptive_max_pool2d(p2, output_size=out_size)

        S_init = (S_p4 + S_p3 + S_p2) / 3
        # S_init = S_p4 + S_p3 + S_p2
        S_weight = self.conv_weight(S_init)
        S_weight = self.softmax(S_weight)
        weight_p4, weight_p3, weight_p2 = torch.split(S_weight, 1, dim=1)
        S_p4_weight = S_p4 * weight_p4
        S_p3_weight = S_p3 * weight_p3
        S_p2_weight = S_p2 * weight_p2
        S = S_p4_weight + S_p3_weight + S_p2_weight

        S_p4 = F.adaptive_max_pool2d(S, output_size=e4.shape[-2:])
        S_p4 = self.S4(S_p4)
        F_p4 = S_p4 + p4
        S_p3 = F.adaptive_max_pool2d(S, output_size=e3.shape[-2:])
        S_p3 = self.S3(S_p3)
        F_p3 = S_p3 + p3
        S_p2 = F.interpolate(S, size=e2.shape[-2:], mode='nearest')
        S_p2 = self.S2(S_p2)
        F_p2 = S_p2 + p2

        b, c, h1, w1 = F_p4.size()
        F_p4 = torch.reshape(F_p4, (b, c, h1 * w1))
        F_p4 = F_p4.permute((0, 2, 1))
        _, _, h2, w2 = F_p3.size()
        F_p3 = torch.reshape(F_p3, (b, c, h2 * w2))
        F_p3 = F_p3.permute((0, 2, 1))
        _, _, h3, w3 = F_p2.size()
        F_p2 = torch.reshape(F_p2, (b, c, h3 * w3))
        F_p2 = F_p2.permute((0, 2, 1))

        Fp = torch.cat((F_p4, F_p3, F_p2), dim=1)
        category = torch.cat([self.scale_p4_embedding, self.scale_p3_embedding, self.scale_p2_embedding], dim=1)
        category = category.cuda()
        Fp = Fp + self.pos_embed + category
        Fp = self.transformer(Fp)

        Fp = Fp.permute((0, 2, 1))
        F_p4, F_p3, F_p2 = Fp[:, :, :9], Fp[:, :, 9:45], Fp[:, :, 45:]
        F_p4 = torch.reshape(F_p4, (b, c, h1, w1))
        F_p3 = torch.reshape(F_p3, (b, c, h2, w2))
        F_p2 = torch.reshape(F_p2, (b, c, h3, w3))

        F_p4 = self.avg_pool4(F_p4)
        F_p4 = F_p4.view(F_p4.size(0), -1)
        F_p4 = self.weight4(F_p4)
        F_p4 = self.fc4(F_p4)
        # F_p4 = self.sigmoid4(F_p4)

        F_p3 = self.avg_pool3(F_p3)
        F_p3 = F_p3.view(F_p3.size(0), -1)
        F_p3 = self.weight3(F_p3)
        F_p3 = self.fc3(F_p3)
        # F_p3 = self.sigmoid3(F_p3)

        F_p2 = self.avg_pool2(F_p2)
        F_p2 = F_p2.view(F_p2.size(0), -1)
        F_p2 = self.weight2(F_p2)
        F_p2 = self.fc2(F_p2)
        # F_p2 = self.sigmoid2(F_p2)

        return output, F_p4, F_p3, F_p2
