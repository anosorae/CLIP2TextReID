from collections import OrderedDict
import math
from typing import Tuple, Union

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

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
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.,):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path((self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, drop_path=dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int, drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.input_resolution = input_resolution # (384, 128)
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5 # 1/sqrt(768)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # add pos drop module from transreid model
        x = self.pos_drop(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        v_global = x[:, 0, :] @ self.proj
    
        return v_global





class CLIP2TextReID(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 # vision
                 image_size: Union[Tuple[int, int], int] =(384, 128),
                 vision_layers: Union[Tuple[int, int, int, int], int]=12,
                 vision_width: int = 768,
                 patch_size: int = 32,
                 stride_size: int = 32,
                 # text
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_layers: int = 12,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.
                 ):
        super().__init__()

        self.context_length = context_length
        self.transformer_width =transformer_width

        # Vision Tranformer init
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_size,
            patch_size=patch_size,
            stride_size=stride_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        # Text transformer init
        transformer_heads = transformer_width // 64
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            drop_path_rate=drop_path_rate
        )

        self.vocab_size = torch.LongTensor(vocab_size)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)

        x = self.pos_drop(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        eos_indices = text.argmax(dim=-1)

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        t_globel = x[torch.arange(x.shape[0]), eos_indices] @ self.text_projection

        return t_globel


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        return image_features, text_features
    
    
    def load_param(self, model_path, jit=False):
        # param_dict = torch.jit.load(model_path, map_location='cpu')
        try:
        # loading JIT archive
            pretrained_model = torch.jit.load(model_path, map_location="cpu")
            pretrained_dict = None
        except RuntimeError:
        # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            pretrained_dict = torch.load(model_path, map_location="cpu")

        pretrained_dict = pretrained_dict or pretrained_model.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        param_dict =  {k: v for k, v in pretrained_dict.items() if k in self.state_dict()}
        # print(param_dict.keys())
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_text_pos_embed(v, self.context_length)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def resize_text_pos_embed(posemb, length):
    old_h, old_w = posemb.shape
    print(f'Resized position embedding from size:{old_h} * {old_w} to size: {length} * {old_w}')

    posemb = posemb.reshape(1, 1, old_h, old_w)  # [1, 1, 77, 512]
    posemb = F.interpolate(posemb, length, mode='bilinear')

    return posemb.squeeze(0)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)



