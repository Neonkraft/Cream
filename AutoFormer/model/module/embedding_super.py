import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np

class PatchembedSuper(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False):
        super(PatchembedSuper, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = embed_dim
        self.scale = scale

    # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        if self.scale:
            return x * self.sampled_scale
        return x
    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops

class WeightEntangledPatchembed(nn.Module):
    def __init__(self, patchembed: PatchembedSuper, embed_dims: list[int]):
        super().__init__()
        self.patchembed = patchembed
        self.embed_dims = embed_dims

    def _compute_mixed_weight(self, weight, embed_dims, alphas):
        max_out_dim = weight.shape[0]
        mixed_weight = 0 + weight

        for embed_dim, alpha in zip(embed_dims, alphas):
            pad_width = max_out_dim - embed_dim
            sub_weight = weight[:embed_dim, ...]
            sub_weight = F.pad(sub_weight, (0, 0, 0, 0, 0, 0, 0, pad_width), 'constant', 0)
            mixed_weight += alpha * sub_weight

        return mixed_weight

    def _compute_mixed_bias(self, bias, embed_dims, alphas):
        if bias is None:
            return None

        max_out_dim = bias.shape[0]
        mixed_bias = 0 + bias

        for embed_dim, alpha in zip(embed_dims, alphas):
            pad_width = max_out_dim - embed_dim
            sub_bias = bias[:embed_dim]
            sub_bias = F.pad(sub_bias, (0, pad_width), 'constant', 0)
            mixed_bias += alpha * sub_bias

        return mixed_bias

    def forward(self, x, alphas):
        mixed_weight = self._compute_mixed_weight(self.patchembed.proj.weight, self.embed_dims, alphas)
        mixed_bias = self._compute_mixed_bias(self.patchembed.proj.bias, self.embed_dims, alphas)

        self.patchembed.sampled_weight = mixed_weight
        self.patchembed.sampled_bias = mixed_bias

        out = self.patchembed(x)

        return out
