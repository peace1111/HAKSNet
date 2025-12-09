import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
#from ..builder import ROTATED_BACKBONES
from mmrotate.models.builder import ROTATED_BACKBONES

from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_,CondConv2d
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer
from torch.nn import functional as F
import numpy
# from timm.models.layers import SelectAdaptivePool2d, Linear, CondConv2d, hard_sigmoid, make_divisible, DropPath


class RSblock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            hidden_features = hidden_features or in_features//4
            out_features = out_features or in_features
            self.dim_conv = in_features // 4
            self.dim_untouched = in_features - self.dim_conv
            self.linear1 = nn.Conv2d(in_features, hidden_features*2 , kernel_size=1)
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
            self.activation = act_layer()
            self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size=3, stride=1, padding=1, bias=False)
            self.linear2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
    def forward(self, x): 
            x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            x = torch.cat((x1, x2), dim=1)
            x1, x2 = self.linear1(x).chunk(2, dim=1)
            x = self.dwconv(x1) * x2
            x = self.linear2(x)
            return x
    

class Channel_attention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)  
        y = self.fc(y) 
        return x * y
    
class SEExpansion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channel_attention = Channel_attention(dim)
        self.cfam = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
    def forward(self, x): 
        y = self.channel_attention(x)
        x = x * y
        return self.cfam(x)
    
class KSSblock(nn.Module):
    def __init__(self, dim, stage_i: int, num_images=10833):
        super().__init__()
        self.dw_conv3  = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dw_conv5  = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw_conv7  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.dw_conv9  = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)

        self.spatial_fusion = nn.Conv2d(4, 4, 7, padding=3)  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.kernel_attn = nn.Sequential(
            nn.Conv2d(dim * 4, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 1),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        U1 = self.dw_conv3(x)
        U2 = self.dw_conv5(U1)
        U3 = self.dw_conv7(U2)
        U4 = self.dw_conv9(U3)

        S1 = torch.mean(U1, dim=1, keepdim=True)
        S2 = torch.mean(U2, dim=1, keepdim=True)
        S3 = torch.mean(U3, dim=1, keepdim=True)
        S4 = torch.mean(U4, dim=1, keepdim=True)

        S_cat = torch.cat([S1, S2, S3, S4], dim=1)
        A_s = self.spatial_fusion(S_cat).sigmoid()  

        U1_s = U1 * A_s[:, 0:1]
        U2_s = U2 * A_s[:, 1:2]
        U3_s = U3 * A_s[:, 2:3]
        U4_s = U4 * A_s[:, 3:4]

        K1 = self.avg_pool(U1)
        K2 = self.avg_pool(U2)
        K3 = self.avg_pool(U3)
        K4 = self.avg_pool(U4)

        K_cat = torch.cat([K1, K2, K3, K4], dim=1)
        A_k = self.kernel_attn(K_cat)  
        Uks = (
            U1_s * A_k[:, 0:1] +
            U2_s * A_k[:, 1:2] +
            U3_s * A_k[:, 2:3] +
            U4_s * A_k[:, 3:4]
        )

        return x * Uks


class Atten(nn.Module):
    def __init__(self, d_model,stage_i = None):
        super().__init__()
        self.spatial_mix = KSSblock(d_model,stage_i)
        self.channe_mix = SEExpansion(d_model)

    def forward(self, x):
        shorcut = x.clone()
        x1 = self.spatial_mix(x)
        x2 = self.channe_mix(x)
        x = x1 + x2
        x = x + shorcut
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, depth = None,norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)

        self.stage_i = depth

        self.attn = Atten(dim,self.stage_i)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = RSblock(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2        
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

@ROTATED_BACKBONES.register_module()
class HAKSNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None,
                 num_classes=15):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        global_depth = 0

        for i in range(num_stages):
            # print(i)
            stage_i = sum(depths[:i])
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([
                Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg,depth=global_depth + j)
                for j in range(depths[i])])
            global_depth += depths[i]
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(HAKSNet, self).init_weights()
            
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


