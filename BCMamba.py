import re
from timm.models.layers import trunc_normal_, DropPath,to_2tuple
from torch.fx.experimental.unification.utils import freeze
import math
from torch.nn import Upsample
from torchvision import models
from .SwinUMambaD import VMamba,UNetResDecoder,TIF
from .MedMamba import MedMamba,medmamba_tiny
from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class LoRALinear(nn.Module):
    """
    LoRA for Linear-like layers.
    - in_features: input dim
    - out_features: output dim
    - rank: LoRA rank r
    - alpha: scaling factor (alpha/r applied to update)
    - bias: whether to keep bias term
    Notes:
      - We keep parameter names 'weight' and 'bias' so pretrained state_dict can load.
      - LoRA params are lora_A (r, in) and lora_B (out, r).
      - Low-rank update: delta_W = lora_B @ lora_A  -> shape (out, in)
      - Final forward: F.linear(x, weight, bias) + F.linear(x, delta_W)
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = rank * 2
        self.scaling = rank * 2 / max(1, rank)
        # main weight and bias (kept, can be loaded from pretrained)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # LoRA parameters: A: (r, in), B: (out, r)
        # we store A as (r, in) and B as (out, r) to compute B @ A -> (out, in)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        # initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x shape: (..., in_features) - we apply linear on last dim
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
            out = out + F.linear(x, delta, None)
        return out

# -------------------------
# ConvNeXt Block / ConvNeXt
# -------------------------
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, rank=0, lora_alpha=1.0):
        """
        dim: channels
        rank: LoRA rank for the pw linear layers (set 0 to disable LoRA in block)
        lora_alpha: alpha scaling
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # use LoRALinear to replace Linear
        self.pwconv1 = LoRALinear(dim, 4 * dim, rank=rank, alpha=lora_alpha, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = LoRALinear(4 * dim, dim, rank=rank, alpha=lora_alpha, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim))) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # to channels_last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # back to channels_first
        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3,3,9,3], dims=[96,192,384,768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., lora_rank=0, lora_alpha=1.0):
        """
        lora_rank: rank used for all LoRALinear layers inside Blocks and head
        if lora_rank == 0 -> LoRA disabled (regular Linear behavior but still uses LoRALinear impl)
        """
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur+j],
                        layer_scale_init_value=layer_scale_init_value,
                        rank=lora_rank,
                        lora_alpha=lora_alpha) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = LoRALinear(dims[-1], num_classes, rank=lora_rank, alpha=lora_alpha, bias=True)

        # init weights for modules that use conventional names
        self.apply(self._init_weights)
        # head scaling
        if isinstance(self.head.bias, torch.nn.Parameter):
            try:
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)
            except Exception:
                pass

    def _init_weights(self, m):
        # only apply to modules that have weight attr named 'weight'
        if isinstance(m, (nn.Conv2d, nn.Linear, LoRALinear)):
            # For LoRALinear, m.weight exists
            try:
                trunc_normal_(m.weight, std=.02)
            except Exception:
                pass
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        skip = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            skip.append(x)
        # global avg pool
        x = self.norm(x.mean([-2, -1]))
        return x, skip

    def forward(self, x):
        x, skip = self.forward_features(x)
        x = self.head(x)
        return x, skip

# -------------------------
# Helper utilities
# -------------------------
def freeze_pretrained_weights_only_lora(model):
    """
    Freeze everything except LoRA parameters (parameters with 'lora_A' or 'lora_B' in their name).
    Use this after loading pretrained weights.
    """
    for name, p in model.named_parameters():
        if ("lora_A" in name) or ("lora_B" in name):
            p.requires_grad = True
        else:
            p.requires_grad = False

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n,p in model.named_parameters() if ("lora_A" in n) or ("lora_B" in n))
    print(f"Total Params:     {total/1e6:.9f} M")
    print(f"Trainable Params: {trainable/1e6:.9f} M")
    print(f"LoRA Params:      {lora/1e6:.9f} M")
    return total, trainable, lora

# -------------------------
# register model factory for timm-style loader
# -------------------------
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=True, in_22k=False, lora_rank=0, lora_alpha=1.0, **kwargs):
    model = ConvNeXt(depths=[3,3,9,3], dims=[96,192,384,768], lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        # checkpoint["model"] likely contains convnext weights with keys that match our param names:
        # since LoRALinear defines 'weight' and 'bias', pretrained linear weights should match.
        try:
            model.load_state_dict(checkpoint["model"], strict=False)
            print("Loaded pretrained checkpoint (strict=False).")
        except Exception as e:
            print("Warning: failed to strictly load pretrained checkpoint:", e)
    return model

class AxialDW(nn.Module):
    """Axail Conv module (1 ×7 and 7 ×1）"""
    def __init__(self, dim, mixer_kernel, dilation=1):#dilation是
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.pw = nn.Conv2d(in_c+out_c, out_c, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Upsample(scale_factor=2)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw1 = AxialDW(out_c, mixer_kernel=(7, 7),dilation=1)
        self.dw2 = AxialDW(out_c, mixer_kernel=(7, 7),dilation=3)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        # x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))

        x = self.bn(self.pw(x))
        x = self.act(self.bn(self.pw2(self.dw1(x)+self.dw2(x))))
        return x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class X_spatial(nn.Module):
    def __init__(self, mamba_chans, cnn_chans):
        super(X_spatial, self).__init__()
        self.cnn_conv = nn.Conv2d(in_channels = cnn_chans, out_channels = mamba_chans, kernel_size=(5, 5), stride = 1, padding = 2)
        self.trans_conv = nn.Conv2d(in_channels = mamba_chans, out_channels = cnn_chans, kernel_size=(3, 3), stride = 1, padding = 1)
        self.out = nn.Conv2d(in_channels=cnn_chans + mamba_chans, out_channels= mamba_chans, kernel_size=(3, 3), stride=1, padding = 1)
    def forward(self, trans, cnn):
        cnn_branch_fuse = cnn + self.trans_conv(trans)
        trans_branch_fuse = trans + self.cnn_conv(cnn)
        combine = torch.cat((cnn_branch_fuse, trans_branch_fuse), dim = 1)
        out = self.out(combine)
        return out


class PyramidPoolUnit(nn.Module):
    """ Pyramid Pooling Unit (PPU) """
    def __init__(self, in_channels, pool_sizes=[1, 3, 6]):
        super(PyramidPoolUnit, self).__init__()
        self.pooling = nn.ModuleList([nn.AdaptiveAvgPool2d(size) for size in pool_sizes])  #  [1,3,6] 池化后维度是多少[B,C,H,W]
        self.conv = nn.Conv2d(in_channels * (len(pool_sizes) + 1), in_channels, kernel_size=1)

    def forward(self, x):
        pooled_features = [F.interpolate(pool(x), size=x.shape[2:], mode='bilinear', align_corners=False) for pool in
                           self.pooling]  #  [B,C,H,W]
        return self.conv(torch.cat([x] + pooled_features, dim=1))   # 768 8 8


class MCFFFD(nn.Module):
    def __init__(self, in_channels, in_channel_list=None,deep_supervised=True):
        super(MCFFFD, self).__init__()
        if in_channel_list is None:
            in_channel_list = [96, 192, 384]
        self.up = Upsample(scale_factor=4)
        self.PPU = PyramidPoolUnit(in_channels)
        self.decoder3= DecoderBlock(768,384)
        self.decoder2 = DecoderBlock(384,192)
        self.decoder1 = DecoderBlock(192,96)
        self.final_conv = nn.Conv2d(96, 1, kernel_size=1)
        self.deep_supervised = deep_supervised
        self.seg_outs = nn.ModuleList([
            nn.Conv2d(ch, 1, 1, 1) for ch in in_channel_list])

    def forward(self, features):
        # sdu_out = []
        # for i,s in enumerate(self.SDU):
        #     sdu_out.append(features[i])
            # f = self.SDU[i](features[i])
            # sdu_out.append(f)  #[1, 96, 64, 64]), torch.Size([1, 192, 32, 32]), torch.Size([1, 384, 16, 16])]

        ppu_out = self.PPU(features[-1])  #768 8 8  self.PPU(features[-1])

        d3 = self.decoder3(ppu_out, features[-2]) #384 16 16
        d2 = self.decoder2(d3, features[-3]) #192 32 32
        d1 = self.decoder1(d2, features[-4]) #96 64 64
        d0 = self.final_conv(self.up(d1))
        seg_output = [d0, d1, d2, d3]
        if self.deep_supervised:
            for i in range(len(self.seg_outs)):
                seg_output[i+1] = self.seg_outs[i](seg_output[i+1])
            return seg_output
        else:
            return d0

"""dual branch network"""
class BCMamba(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth_dim=None,deep_supervised=True, lora_rank=8, lora_alpha=16.0):
        super().__init__()
        if depth_dim is None:
            depth_dim = [96, 192, 384, 768]
        self.cnnnet =  convnext_tiny(pretrained=True, lora_rank=lora_rank, lora_alpha=lora_alpha)
        freeze_pretrained_weights_only_lora(self.cnnnet)
        """Encoder"""
        self.vssm_encoder  = VMamba()
        self.Fuse = nn.ModuleList()
        for i in range(len(depth_dim)):
            self.Fuse.append(X_spatial(depth_dim[i],depth_dim[i]))
        """Decoder"""
        # self.decoder = MCFFFD(768,[96,192,384],deep_supervised=deep_supervised)
        self.deep_supervised = deep_supervised

        self.decoder1 = UNetResDecoder(out_channels,deep_supervised,[96,192,384,768])
        # self.apply(self._init_weights)
        self.return_features = True

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv1d):
    #         n = m.kernel_size[0] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        """Encoder"""
        cnn_x,cnn_out = self.cnnnet(x)  #([1, 96, 64, 64]),([1, 192, 32, 32]),([1, 384, 16, 16]),([1, 768, 8, 8])]
        vss_out = self.vssm_encoder(x) #96 64 64 ,192 32 32 , 384 16 16, 768 8 8

        """fusion module"""
        # res_out = []
        # for i in range(len(self.Fuse)):
        #     if i==0: res_out.append(vss_out[i])
        #     # fuse = self.Fuse[i](vss_out[i+1], cnn_out[i])
        #     fuse = vss_out[i+1] + cnn_out[i]
        #     res_out.append(fuse)

        """Decoder"""
        seg_out =  self.decoder1(vss_out)
        if self.deep_supervised:
            for i,o in enumerate(seg_out):
                seg_out[i] = F.interpolate(o,(256,256),mode='bilinear',align_corners=True)
            # print([o.shape for o in seg_out])
            return seg_out
        else:
            return seg_out

        # if self.return_features:
        #     return seg_out, cnn_out[-2], vss_out[-1], res_out[-1]
        # return seg_out

    # @torch.no_grad()
    # def freeze_encoder(self):
    #     for name, param in self.vssm_encoder.named_parameters():
    #         if "patch_embed" not in name:
    #             param.requires_grad = False
    #
    # @torch.no_grad()
    # def unfreeze_encoder(self):
    #     for param in self.vssm_encoder.parameters():
    #         param.requires_grad = True


class MedFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth_dim=None,deep_supervised=True, lora_rank = 16, lora_alpha = 32.0):
        super().__init__()
        if depth_dim is None:
            depth_dim = [96, 192, 384, 768]
        self.cnnnet = convnext_tiny(pretrained=True, lora_rank=lora_rank, lora_alpha=lora_alpha)
        freeze_pretrained_weights_only_lora(self.cnnnet)

        # self.vmunet  = MedMamba()
        self.vmunet  = medmamba_tiny(rank=lora_rank)
        self.Fuse = nn.ModuleList()
        for i in range(len(depth_dim)):
            self.Fuse.append(TIF(depth_dim[i],depth_dim[i]))
        # self.decoder = MCFFFD(768,[96,192,384],deep_supervised=deep_supervised)
        self.decoder1 = UNetResDecoder(out_channels,deep_supervised,[96,192,384,768])
        self.deep_supervised = deep_supervised
        self.return_features = False
    def forward(self, x):
        """Dual Encoder"""
        cnn_x,cnn_out = self.cnnnet(x)  #([1, 96, 64, 64]),([1, 192, 32, 32]),([1, 384, 16, 16]),([1, 768, 8, 8])]
        vss_out = self.vmunet(x) # (96 64 64) , (192 32 32) , (384 16 16), (768 8 8)

        """CNN Encoder"""
        # vss_out = []
        # vss_out.append(x)
        # for i in range(4):
        #     vss_out.append(cnn_out[i])

        """fusion module"""
        res_out = []
        for i in range(len(self.Fuse)):
            if i==0: res_out.append(vss_out[i])
            fuse = self.Fuse[i](vss_out[i+1], cnn_out[i])
            # fuse = vss_out[i+1] + cnn_out[i]
            res_out.append(fuse)
        """Decoder"""
        # seg_out =  self.decoder1(res_out)

        seg_out =  self.decoder1(res_out)
        if self.deep_supervised:
            for i,o in enumerate(seg_out):
                seg_out[i] = F.interpolate(o,(256,256),mode='bilinear',align_corners=True)
            # print([o.shape for o in seg_out])
            return seg_out
        else:
            return seg_out

        # if self.return_features:
        #     return seg_out, cnn_out[-1], vss_out[-1], res_out[-1]

        # return seg_out

    # @torch.no_grad()
    # def freeze_encoder(self):
    #     for name, param in self.vssm_encoder.named_parameters():
    #         if "patch_embed" not in name:
    #             param.requires_grad = False
    #
    # @torch.no_grad()
    # def unfreeze_encoder(self):
    #     for param in self.vssm_encoder.parameters():
    #         param.requires_grad = True


class res(nn.Module):
    def __init__(self,in_channel=3,out_channels=1):
        super(res, self).__init__()
        resnet = models.resnet34(pretrained=True)   #resnet18 11.18M 2.38G    resnet34 21.28M 4.81G
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        x = self.inc(x)
        x = self.pool(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        return x

def load_pretrained_ckpt(
        model,
        num_input_channels=1,
        ckpt_path="./pretrained_ckpt/vmamba_tiny_e292.pth"
):
    print(f"Loading weights from: {ckpt_path}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias"]
    ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)
    model_dict = model.state_dict()
    for k, v in ckpt['model'].items():
        if k in skip_params:
            print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"
        if "patch_embed" in k and ckpt['model']["patch_embed.proj.weight"].shape[1] != num_input_channels:
            print(f"Passing weights: {k}")
            continue
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape},{kr}"
            model_dict[kr] = v
        else:
            print(f"Passing weights: {k}")

    model.load_state_dict(model_dict)
    return model

def load_from(model,ckpt_path="./pretrained_ckpt/Breast_MedMamba.pth"):
    print(f"Loading weights from: {ckpt_path}")
    model_dict = model.state_dict()
    # print(model_dict.keys())   #vmunet.vmunet.layers.0.blocks.0.attn.relative_position_bias_table
    modelCheckpoint = torch.load(ckpt_path, weights_only=False)
    # print(modelCheckpoint.keys())            print(model_dict.keys())
    pretrained_dict = modelCheckpoint
    new_dict = {}
    for k, v in pretrained_dict.items():  # fc.weight,fc.bias
        kr = f"vmunet.vmunet.{k}"
        # kr = kr.replace(f"vmunet", f"")#
        if kr in model_dict.keys():
            if "head" in kr:
                continue
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            new_dict[kr] = v
        else:
            print(f"Passing weights: {k}")
    model_dict.update(new_dict) # 打印出来，更新了多少的参数
    print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                               len(pretrained_dict),
                                                                               len(new_dict)))
    load_partial_state_dict(model, model_dict)  # total 117 pretrained 276 update 4
    print("encoder loaded finished!")

def load_partial_state_dict(model, state_dict):  #
    own_state = model.state_dict()

    for name, param in state_dict.items():  #
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                print(f'Skipping {name} due to size mismatch.')
                print(own_state[name].shape, param.shape)
        else:
            print(f'Skipping {name} as it is not in the model.')



def get_sm_model(rank=4, deep_supervised=True):
    model = BCMamba(lora_rank = rank, deep_supervised=deep_supervised).cuda()
    # model = load_pretrained_ckpt(model)   #57.88M 11.26G
    return model
def get_med_model(rank=4, deep_supervised=True):
    model = MedFormer(lora_rank = rank, deep_supervised=deep_supervised).cuda()
    # load_from(model)
    return model

if __name__ == '__main__':
    model = res(3,1).cuda()
    # load_from(model)
    # x = torch.randn(2,3,256,256).cuda()
    # out = model(x)
    # print([o.shape for o in out])
    macs, params = get_model_complexity_info(model, (3,256,256), as_strings=True, print_per_layer_stat=True)
    print(f"模型 Params and FLOPs:{params}, {macs}")

    # encoder = EncoderBlock(16, 32).cuda()
    # test_input = torch.randn(2, 16, 256, 256).cuda()
    # test_output, _ = encoder(test_input)
    # print(f"编码器输出范围: {test_output.min():.3f} ~ {test_output.max():.3f}")  #  -2.050 ~ 5.666

    # mamba_layer = MambaLayer(dim=16).cuda()
    # test_seq = torch.randn(1, 16, 65536).cuda()  # (batch, seq_len, dim)
    # output = mamba_layer(test_seq)
    # assert not torch.isnan(output).any()
    # print(output.shape)

    #test channel attention
    # model  = CrossAttentionFusion(96).cuda()
    # x = torch.randn(2,96,64,64).cuda()
    # y = torch.randn(2,96,64,64).cuda()
    # out = model(x,y)
    # print(out.shape)

    # test PPU model
    # model = PyramidPoolUnit(768).cuda()
    # x = torch.randn(2, 768, 8, 8).cuda()
    # out  = model(x)
    # print(out.shape)

    # test MCFFFD
    # model = MCFFFD(768,[96,192,384]).cuda()
    # x= torch.rand(1,96,64,64).cuda()
    # x1 = torch.rand(1,192,32,32).cuda()
    # x2 = torch.rand(1,384,16,16).cuda()
    # x3 = torch.rand(1,768,8,8).cuda()
    # input = [x,x1,x2,x3]
    # #
    # out  = model(input)
    # print([o.shape for o in out ])
    #
    # model = PVMLayer(192, 384, d_state=16, d_conv=4, expand=2).cuda()
    # x = torch.randn(2, 192, 32, 32).cuda()
    # out = model(x)
    # print(out.shape)
