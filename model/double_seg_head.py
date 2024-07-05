import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import backbone as encoder
from . import decoder
import sys
"""
Borrow from https://github.com/facebookresearch/dino
"""
sys.path.append('')

class CTCHead(nn.Module):
    #cls-token投影
    def __init__(self, in_dim, out_dim=4096, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # pdb.set_trace()
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class Fmap_proj_Head(nn.Module):
    #cls-token投影
    def __init__(self, in_dim, out_dim=4096, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=2048):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # pdb.set_trace()
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
}


class Mlp(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_classes=20):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_classes=num_classes)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, mask_type=None,aux_layer = -3):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.mask_type = mask_type
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self._size = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.aux_layer = aux_layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_classes=num_classes)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def named_parameters(self, recurse=True):
    #     return super().named_parameters(recurse=recurse)
    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        h, w = h // self.patch_embed.patch_size[0], w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)  # patch linear embedding

        patch_pos_embed = self.pos_embed[:, 1:, :].reshape(1, self._size, self._size, -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos_embed = patch_pos_embed.reshape(1, -1, h*w).permute(0, 2, 1)
        pos_embed = torch.cat((self.pos_embed[:, :1, :], patch_pos_embed), dim=1)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + pos_embed

        return x

    def forward_features(self, x, n):
        # B = x.shape[0]
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []
        embeds = []
        for i, blk in enumerate(self.blocks):
            x, weights = blk(x)
            embeds.append(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], attn_weights,embeds[self.aux_layer]

    def forward(self, x, n=12):
        x, attn_weights,_ = self.forward_features(x, n)
        x = self.head(x)

        if self.training:
            return x
        else:
            return x, attn_weights


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 192 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = default_cfgs['vit_tiny_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch16_224']
    return model


@register_model
def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch32_384']
    return model






#single-head————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class network(VisionTransformer):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum

        img_size = to_2tuple(self.img_size)
        self.patch_size = patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)

        self.Fmap_proj_head = Fmap_proj_Head(in_dim=self.embed_dim, out_dim=1024)
    
        # self.proj_head_t = CTCHead(in_dim=self.embed_dim, out_dim=1024,)

        # for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
        #     param_t.data.copy_(param.data)  # initialize teacher with student
        #     param_t.requires_grad = False  # do not update by gradient

        #教师-学生模型是一种常用的知识蒸馏（knowledge distillation）方法，用于在训练过程中通过引入教师模型来辅助学生模型的训练

        self.in_channels = self.embed_dim
        #如果 self.encoder 对象或类的实例具有属性 embed_dim，
        # 则将 self.in_channels 设置为长度为4的列表，每个元素都是 self.encoder.embed_dim。
        # 【D,D,D,D】
        # self.pooling = F.adaptive_avg_pool1d
        #?
        self.decoder = decoder.LargeFOV(in_planes=self.in_channels, out_planes=self.num_classes,)
        #                                                 D
        self.classifier = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.crop_classifier = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.fmap_fusion = nn.Linear(in_features=4 * self.in_channels,out_features= self.in_channels ,bias= True)
        # self.token_classifier = nn.Linear(in_features=self.in_channels, out_features=self.num_classes-1,bias=False,)
        #两个CAM（一个辅助的，一个后边儿的）

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self, n_iter=None):
        ## no scheduler here
        #为了更新p-global
        momentum = self.init_momentum
        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;
        param_groups_name = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        param_groups[2].append(self.classifier.weight)
        param_groups_name[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)
        param_groups_name[2].append(self.aux_classifier.weight)
        param_groups_name[2].append(self.crop_classifier.weight)
        param_groups_name[2].append(self.fmap_fusion.weight)
        # param_groups[2].append(self.token_classifier.weight)

        for name, param in list(self.Fmap_proj_head.named_parameters()):
            param_groups[2].append(param)



        for name, param in list(self.decoder.named_parameters()):
            param_groups[3].append(param)


        for name, param in list(self.named_parameters()):
            if id(param) not in (id(p) for p in param_groups[2]) and id(param) not in (id(p) for p in param_groups[3]):
                if "norm" in name:
                    param_groups[1].append(param)
                    param_groups_name[1].append(name)
                else:
                    param_groups[0].append(param)
                    param_groups_name[0].append(name)

        return param_groups

    def to_2D(self, x, h, w):
        #x B N D
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        #b C H W
        return x

    def forward_proj(self, crops, n_iter=None,select_k = 1):
        
        #crops [num * [b 3 cs cs]]
        global_view = crops[:2]
        local_view = crops[2:]
        hg, wg = global_view[0].shape[-2] // self.patch_size[0], global_view[0].shape[-1] // self.patch_size[1]
        hl , wl = local_view[0].shape[-2] // self.patch_size[0], local_view[0].shape[-1] // self.patch_size[1]
        topk = int(hl * wl / 2)
        local_inputs = torch.cat(local_view, dim=0)
        # (num-2)*b 3 CS CS
        #ema更新global
        # self._EMA_update_encoder_teacher(n_iter)
        #[4 768]
        # global_output_t = self.forward_features(torch.cat(global_view, dim=0))[1].detach()
        #cls-token [2*b , dim]
        global_output_patches = self.forward_features(torch.cat(global_view, dim=0))[1]
        global_output_patches= global_cam = self.to_2D(global_output_patches,hg,wg)
        global_output_patches = F.adaptive_max_pool2d(global_output_patches,(1,1))
        output_global_cam = self.crop_classifier(global_output_patches).squeeze(-1).squeeze(-1)
        #cls-token [2*b , cls ,dim_t]
        #return 的 cls-token
        local_output_patches = self.forward_features(local_inputs)[1]
        local_output_patches= local_cam = self.to_2D(local_output_patches,hl,wl)
        local_output_patches = local_output_patches.view(local_output_patches.shape[0],local_output_patches.shape[1],-1).permute(0,2,1)
        sorted_local_output_patches,_ = torch.sort(local_output_patches,-2,descending=True)
        sorted_local_output_patches = torch.mean(sorted_local_output_patches[:,:select_k,:],dim=-2).unsqueeze(-1).unsqueeze(-1)
        # local_output_patches = F.adaptive_max_pool2d(local_output_patches,(1,1))
        output_local_cam = self.crop_classifier(sorted_local_output_patches).squeeze(-1).squeeze(-1)
        

        #[1*8 ,cls,dim]
        # output_local = self.proj_head(local_output_multi)

        return output_global_cam,output_local_cam
        #[2 dim_t]  [8 dim_t]

    def forward_features(self, x, n=12):

        x = self.prepare_tokens(x)

        x = self.pos_drop(x)
        embeds = []
        for blk in self.blocks:
            x, weights = blk(x,)
            embeds.append(x)

        x = self.norm(x)
        embeds[-1] = x
        return x[:, 0], x[:, 1:], embeds[self.aux_layer][:, 1:],embeds

    def forward(self, x, cam_only=False, crops=None, n_iter=None,cam_crop = False,select_k = 1,return_cam = False):
        #x (b c h w) [2 3 448 448]
        cls_token ,_x, x_aux, embeds= self.forward_features(x)
        #cls-token final-patch  mid-patch
        #b 1 D     B N（784） D（768）         B N D
        #crops 应该是裁剪出来的nertural区域
        
        # if crops is not None:
        #     output_t, output_s = self.forward_proj(crops, n_iter,select_k=select_k)


        h, w = x.shape[-2] // self.patch_size[0], x.shape[-1] // self.patch_size[1]

        _x4 = self.to_2D(_x, h, w)
        #B D 28 28
        _x_aux = self.to_2D(x_aux, h, w)
        #B C H W(patch-num)
        seg = self.decoder(_x4) 
        #图片尺寸没有变化，感受野变大了[2 21 28 28]
        if cam_only:

            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
                            # INPUT D OUT C  KERNEL 1*1
            return cam_aux, cam 
        
        if cam_crop:

            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
            cam_crop = F.conv2d(_x4, self.crop_classifier.weight).detach()
                            # INPUT D OUT C  KERNEL 1*1
            return cam_aux, cam ,cam_crop
        #【b 20 28 28】
        
        #生成CAM 并分类提供监督
        #分类   
        #B C D -> B C 1
        # multi_cls_pooling =  torch.mean(multi_cls,dim = -1)
        if return_cam:
            cam_12th = cam = F.conv2d(_x4, self.classifier.weight)
        
        #B N D
        x_aux_cls = _x_aux.view(_x_aux.shape[0],_x_aux.shape[1],-1).permute(0,2,1)
        #B N D
        sorted_x_aux_cls,_ = torch.sort(x_aux_cls,-2,descending=True)
        cls_aux = torch.mean(sorted_x_aux_cls[:,:5,:],dim=-2).unsqueeze(-1).unsqueeze(-1)
        cls_aux = self.aux_classifier(cls_aux)

        x4_cls = _x4.view(_x4.shape[0],_x4.shape[1],-1).permute(0,2,1)
        #B N D
        sorted_x4_cls,_ = torch.sort(x4_cls,-2,descending=True)
        cls_x4 = torch.mean(sorted_x4_cls[:,:5,:],dim=-2).unsqueeze(-1).unsqueeze(-1)
        cls_x4 = self.classifier(cls_x4)
    

        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        cls_aux = cls_aux.view(-1, self.num_classes-1)
            # cls_to_classify = self.token_classifier(cls_token)
        #生成CAM 并分类提供监督
        #分类
        
        if crops is None:
            return cls_x4, seg, _x4, cls_aux
        elif not return_cam:
            return cls_x4, seg, _x4, cls_aux, 
        else:
            return cls_x4, seg, _x4, cls_aux, cam_12th

def get_pertrained_dict(pretrain_path):
    trained_state_dict = torch.load(pretrain_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if 'model' in trained_state_dict:
        model_state_dict = trained_state_dict['model']
        for k, v in model_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
    else:
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
    return new_state_dict

class network_du_heads_independent_config(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None, pretrained_path= None):
        super(network_du_heads_independent_config, self).__init__()

        self.branch1 = network(backbone, num_classes, pretrained, init_momentum, aux_layer)
        self.branch2 = network(backbone, num_classes, pretrained, init_momentum, aux_layer)
        
        b1_pretrained_path = '/home/zhonggai/python-work-space/DEFormer/DEFormer/jx_vit_base_p16_224-80ecf9dd.pth'
        b2_pretrained_path = '/home/zhonggai/python-work-space/DEFormer/DEFormer/jx_vit_base_p16_224-80ecf9dd.pth'
        b1_pretrained_dict = get_pertrained_dict(b1_pretrained_path)
        b2_pretrained_dict = get_pertrained_dict(b2_pretrained_path)
        
        self.branch1.load_state_dict(b1_pretrained_dict, strict=False)
        self.branch2.load_state_dict(b2_pretrained_dict, strict=False)

    def get_param_groups(self):
        b1_param_groups = self.branch1.get_param_groups()
        b2_param_groups = self.branch2.get_param_groups()
        param_groups = [b1_param_groups[0]+b2_param_groups[0],b1_param_groups[1]+b2_param_groups[1],b1_param_groups[2]+b2_param_groups[2],b1_param_groups[3]+b2_param_groups[3]]
        return param_groups
        
    def eval_branch(self,branch):
        if branch == 'b1':
            return self.branch1
        else:
            return self.branch2

    def forward(self, x, cam_only=False, crops=None, n_iter=None,cam_crop = False,select_k = 1,return_cam = False):
        if cam_only == True:
            b1_cam_aux, b1_cam  = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
            b2_cam_aux, b2_cam  = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
            return (b1_cam_aux, b2_cam_aux), (b1_cam, b2_cam)
        
        else:
            if crops is None:
                b1_cls_x4, b1_seg, b1_x4, b1_cls_aux = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                b2_cls_x4, b2_seg, b2_x4, b2_cls_aux = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                return (b1_cls_x4,b2_cls_x4),(b1_seg,b2_seg), (b1_x4, b2_x4), (b1_cls_aux, b2_cls_aux)
            elif not return_cam:
                b1_cls_x4, b1_seg, b1_x4, b1_cls_aux = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                b2_cls_x4, b2_seg, b2_x4, b2_cls_aux = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)            
                return (b1_cls_x4,b2_cls_x4),(b1_seg,b2_seg), (b1_x4, b2_x4), (b1_cls_aux, b2_cls_aux)
            else:
                b1_cls_x4, b1_seg, b1_x4, b1_cls_aux, b1_cam12th = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                b2_cls_x4, b2_seg, b2_x4, b2_cls_aux, b2_cam12th = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)  
                return (b1_cls_x4,b2_cls_x4),(b1_seg,b2_seg), (b1_x4, b2_x4), (b1_cls_aux, b2_cls_aux), (b1_cam12th, b2_cam12th) 
            



class network_du_heads_shared_config(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None, pretrained_path= None):
        super(network_du_heads_shared_config, self).__init__()
        shared_network = network(backbone, num_classes, pretrained, init_momentum, aux_layer)
        self.branch1 = shared_network
        self.branch2 = shared_network
        
        pretrained_path = '/home/zhonggai/python-work-space/DEFormer/DEFormer/jx_vit_base_p16_224-80ecf9dd.pth'
        network_pertrained_dict = get_pertrained_dict(pretrained_path)
        shared_network.load_state_dict(network_pertrained_dict, strict=False)

    def get_param_groups(self):
        param_groups = self.branch1.get_param_groups()

        return param_groups
    
    def eval_branch(self,branch):
        if branch == 'b1':
            return self.branch1
        else:
            return self.branch2
    

    def forward(self, x, cam_only=False, crops=None, n_iter=None,cam_crop = False,select_k = 1,return_cam = False):
        if cam_only == True:
            b1_cam_aux, b1_cam  = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
            b2_cam_aux, b2_cam  = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
            return (b1_cam_aux, b2_cam_aux), (b1_cam, b2_cam)
        
        else:
            if crops is None:
                b1_cls_x4, b1_seg, b1_x4, b1_cls_aux = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                b2_cls_x4, b2_seg, b2_x4, b2_cls_aux = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                return (b1_cls_x4,b2_cls_x4),(b1_seg,b2_seg), (b1_x4, b2_x4), (b1_cls_aux, b2_cls_aux)
            elif not return_cam:
                b1_cls_x4, b1_seg, b1_x4, b1_cls_aux = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                b2_cls_x4, b2_seg, b2_x4, b2_cls_aux = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)            
                return (b1_cls_x4,b2_cls_x4),(b1_seg,b2_seg), (b1_x4, b2_x4), (b1_cls_aux, b2_cls_aux)
            else:
                b1_cls_x4, b1_seg, b1_x4, b1_cls_aux, b1_cam12th = self.branch1.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)
                b2_cls_x4, b2_seg, b2_x4, b2_cls_aux, b2_cam12th = self.branch2.forward(x, cam_only, crops, n_iter, cam_crop, select_k, return_cam)  
                return (b1_cls_x4,b2_cls_x4),(b1_seg,b2_seg), (b1_x4, b2_x4), (b1_cls_aux, b2_cls_aux), (b1_cam12th, b2_cam12th) 
            
