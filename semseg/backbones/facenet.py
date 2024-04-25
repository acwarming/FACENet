import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
import functools
from functools import partial
from fvcore.nn import flop_count_table, FlopCountAnalysis
from semseg.models.modules.fcem import FCEM
from semseg.models.modules.fcem import ChannelEmbed
from semseg.models.modules.mspa import MSPABlock
from semseg.utils.utils import nchw_to_nlc, nlc_to_nchw

class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
      
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1, -2)
        _, _, Nk, Ck = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))

        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(
            1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]

class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2).view(b, c)
        return y

class Feature_MaxPool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_MaxPool, self).__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2).view(b, c)
        return y


class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim // ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // ratio, dim)

    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gmp_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        return max_out


class Channel_Attention_Max(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention_Max, self).__init__()
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim // ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // ratio, dim)

    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gmp_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        return max_out

class Channel_Attention_Avg(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention_Avg, self).__init__()
        self.gmp_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim // ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // ratio, dim)

    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gmp_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        return max_out


class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1

# wait
class MMFM(nn.Module):

facenet_settings = {
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class FACENet(nn.Module):
    def __init__(self, model_name: str = 'B4', modals: list = ['rgb', 'ther']):
        super().__init__()
        assert model_name in face_settings.keys(), f"Model name should be in {list(face_settings.keys())}"
        embed_dims, depths = face_settings[model_name]
        extra_depths = depths
        self.modals = modals[1:] if len(modals) > 1 else []
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        norm_cfg = dict(type='BN', requires_grad=True)

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7 // 2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        if self.num_modals > 0:
            self.FCEMs = nn.ModuleList([
                FCEM(dim=embed_dims[0], reduction=1),
                FCEM(dim=embed_dims[1], reduction=1),
                FCEM(dim=embed_dims[2], reduction=1),
                FCEM(dim=embed_dims[3], reduction=1)])

        self.mmfm0 = MMFM(64)
        self.mmfm1 = MMFM(128)
        self.mmfm2 = MMFM(320)
        self.mmfm3 = MMFM(512)

    def forward(self, x: list) -> list:
        x_cam = x[0]
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]
        outs = []
        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x_t, H, W = self.patch_embed1(x_ext[0])
        for blk in self.block1:
            x_t = blk(x_t, H, W)
        x1_t = self.norm1(x_t).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x1_cam, x1_t = self.FCEMs[0](x1_cam, x1_t)
        x_f = torch.cat([x1_cam, x1_t], dim=1)
        x1_cam, x1_t = self.mmfm0(x_f)
        outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x_t, H, W = self.patch_embed2(x1_t)
        for blk in self.block2:
            x_t = blk(x_t, H, W)
        x2_t = self.norm2(x_t).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x2_cam, x2_t = self.FCEMs[1](x2_cam, x2_t)
        x_f = torch.cat([x2_cam, x2_t], dim=1)
        x2_cam, x2_t = self.mmfm1(x_f)
        outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x_t, H, W = self.patch_embed3(x2_t)
        for blk in self.block3:
            x_t = blk(x_t, H, W)
        x3_t = self.norm3(x_t).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x3_cam, x3_t = self.FCEMs[2](x3_cam, x3_t)
        x_f = torch.cat([x3_cam, x3_t], dim=1)
        x3_cam, x3_t = self.mmfm2(x_f)
        outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x_t, H, W = self.patch_embed4(x3_t)
        for blk in self.block4:
            x_t = blk(x_t, H, W)
        x4_t = self.norm4(x_t).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x4_cam, x4_t = self.FCEMs[3](x4_cam, x4_t)
        x_f = torch.cat([x4_cam, x4_t], dim=1)
        x4_cam, x4_t = self.mmfm3(x_f)
        outs.append(x4_cam)

        return outs


if __name__ == '__main__':
    modals = ['img', 'ther']
    x = [torch.zeros(1, 3, 480, 640), torch.ones(1, 1, 480, 640)]
    model = FACENet('B4', modals)
    outs = model(x)
    for y in outs:
        print(y.shape)
