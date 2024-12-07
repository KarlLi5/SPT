# -----------------------------------------------------------------------------------
# Scale-aware Backprojection Transformer (SPT)
# -----------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class fused_cab(nn.Module):
    def __init__(self, dim=96):
        super(fused_cab, self).__init__()
        self.cab = CAB(num_feat=dim, compress_ratio=6, squeeze_factor=16)
        self.conv_1 = nn.Conv2d(dim, dim, 1)
        self.conv_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, shortcut, x):
        cab_shortcut = self.cab(shortcut)
        x_sub = self.conv_1(cab_shortcut - x)
        x_out = x_sub + self.conv_2(x)

        return x_out


class FFN(nn.Module):
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


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=6, squeeze_factor=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class PPSA(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[2, 4, 8]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim  # dim == Channel
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios

        self.norm = nn.LayerNorm(dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, stage, d_convs=None):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

        pools = []

        # pool
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool = F.adaptive_max_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio))) + pool
            pool = pool + l(pool)  # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))
        pools = torch.cat(pools, dim=2).permute(0, 2, 1)
        pools = self.norm(pools)

        # self_attention
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SPAL(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 FFN_ratio=2., cab_scale=1., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, stage=0, layer=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.FFN_ratio = FFN_ratio
        self.stage = stage
        self.layer = layer

        self.norm1 = norm_layer(dim)
        self.up_dim = 128
        self.conv_in = nn.Conv2d(dim, self.up_dim, 1)
        self.CAB = CAB(num_feat=self.up_dim // 2, compress_ratio=3, squeeze_factor=16)
        self.attn = PPSA(self.up_dim // 2, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                         proj_drop=0., pool_ratios=[2, 4, 8])
        self.conv_up = nn.Conv2d(self.up_dim // 2, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cab_scale = cab_scale
        self.conv = nn.Conv2d(self.up_dim // 2, dim, 1)
        self.norm2 = norm_layer(dim)
        FFN_hidden_dim = int(dim * FFN_ratio)
        self.FFN = FFN(in_features=dim, hidden_features=FFN_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, d_convs=None):
        H, W = x_size
        _, _, C = x.shape
        shortcut = x

        x_ = self.norm1(x)
        x_spa, x_cab = np.split(self.conv_in(x_.reshape(-1, H, W, C).permute(0, 3, 1, 2)), 2, axis=1)

        x_spa = self.attn(x_spa.permute(0, 2, 3, 1).reshape(-1, H * W, self.up_dim // 2), W, H, self.stage, d_convs)
        x_spa_1 = x_spa.reshape(-1, H, W, self.up_dim // 2).permute(0, 3, 1, 2)
        x_cab = self.CAB(x_cab)

        x_sub = self.conv(x_cab - x_spa_1)
        x = shortcut + (self.conv_up(x_spa_1) + x_sub * self.cab_scale).permute(0, 2, 3, 1).reshape(-1, H * W, self.dim)

        x = x + self.drop_path(self.FFN(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"FFN_ratio={self.FFN_ratio}"


class SPTB(nn.Module):
    def __init__(self, dim, input_resolution, SPTB_num, num_heads,
                 FFN_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, img_size=224, patch_size=4,
                 downsample=None, use_checkpoint=False, stage=0, SPAL_num=3):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.SPAL_num = SPAL_num
        self.SPTB_num = SPTB_num
        self.use_checkpoint = use_checkpoint

        # build Block
        self.Self_PPSA_Block = nn.ModuleList([
            SPAL(dim=dim, input_resolution=input_resolution,
                                             num_heads=num_heads,
                                             FFN_ratio=FFN_ratio,
                                             cab_scale=1,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path,
                                             norm_layer=norm_layer, stage=stage, layer=i)
            for i in range(SPAL_num)])

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        # Hierarchical Backprojection Learning
        self.conv_split = nn.Conv2d(dim, dim * 2, 1)
        self.conv_sum = nn.Conv2d(dim, dim, 3, 1, 1)
        self.fused = fused_cab(self.dim)

    def forward(self, x, x_size, d_convs=None):
        shortcut = x
        _, _, C = x.shape

        # split
        x_spa, x_cab = np.split(self.conv_split(x.reshape(-1, x_size[0], x_size[1], C).permute(0, 3, 1, 2)), 2, axis=1)
        x_spa = x_spa.flatten(2).transpose(1, 2)

        # deep feature learning
        for blk in self.Self_PPSA_Block:
            x_spa = blk(x_spa, x_size, d_convs)

        # Hierarchical Backprojection Learning
        x_spa = self.conv_sum(self.patch_unembed(x_spa, x_size))
        x_out = self.fused(x_cab, x_spa)
        x_output = self.patch_embed(x_out) + shortcut

        return x_output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, SPAL_num={self.SPAL_num}"

class SPT(nn.Module):
    def __init__(self, img_size=None, patch_size=1, in_chans=3,
                 embed_dim=96, SPTB_num=6, SPAL_num=3, num_head=6,
                 FFN_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(SPT, self).__init__()
        if img_size is None:
            img_size = [128, 128]
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        ################################### 2, deep feature extraction ######################################
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.FFN_ratio = FFN_ratio
        self.SPTB_num = SPTB_num
        self.pool_ratios = [2, 4, 8]

        # patch embed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # patch unEmbed
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # position encoding
        dconv_dim = 64
        self.d_convs = nn.ModuleList(
            [nn.Conv2d(dconv_dim, dconv_dim, kernel_size=3, stride=1, padding=1, groups=dconv_dim) for temp in self.pool_ratios])

        # build SPTB
        self.SPAL_num = SPAL_num
        self.layers = nn.ModuleList()
        for i_layer in range(SPTB_num):
            layer = SPTB(dim=embed_dim,
                                input_resolution=(patches_resolution[0], patches_resolution[1]),
                                SPTB_num=SPTB_num,
                                num_heads=num_head,
                                FFN_ratio=self.FFN_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=0.,
                                img_size=img_size,
                                patch_size=patch_size,
                                norm_layer=norm_layer,
                                use_checkpoint=use_checkpoint,
                                stage=i_layer,
                                SPAL_num=self.SPAL_num)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the task conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

            ################################ 4, reconstructive backprojection learning ################################
            self.final_feat1 = nn.Conv2d(num_out_ch, num_feat, 3, 1, 1)
            self.final_feat2 = nn.Conv2d(num_feat, num_out_ch, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        # embed
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # deep feature learning
        for layer in self.layers:
            x = layer(x, x_size, self.d_convs)

        # unembed
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]

        shortcut1 = x

        x_hr = F.interpolate(x, scale_factor=self.upscale, mode='bilinear')

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)

            # upsample
            x = self.conv_last(self.upsample(x)) + x_hr

            # backprojection learning
            LR_res = shortcut1 - F.interpolate(x, scale_factor=1/self.upscale, mode='bilinear')
            LR_res = self.final_feat1(LR_res)
            LR_res = self.final_feat2(LR_res)
            SR_res = F.interpolate(LR_res, scale_factor=self.upscale, mode='bilinear')
            x = x + SR_res

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]
