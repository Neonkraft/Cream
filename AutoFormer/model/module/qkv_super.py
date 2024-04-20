import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.module.lora_layers import LoraLinear

class qkv_super(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops

def sample_weight(weight, sample_in_dim, sample_out_dim):

    sample_weight = weight[:, :sample_in_dim]
    sample_weight = torch.cat([sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim =0)

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias

class WeightEntangledLoraQKV(nn.Module):
    def __init__(self, qkv, in_feature_dims: list[int], n_head_choices: list[int]):
        super().__init__()
        self.qkv = LoraLinear(qkv)
        self.in_feature_dims = in_feature_dims
        self.n_head_choices = n_head_choices

    def activate_lora(self, *args, **kwargs):
        self.qkv.activate_lora(*args, **kwargs)

    # def set_arch_weights(self, embed_weights, n_heads_weights):
    #     self.embed_weight = embed_weights
    #     self.n_heads_weight = n_heads_weights

    def _compute_combined_weight(self, weight, in_feature_dims, n_head_choices, alphas, betas):
        max_out_dim, max_in_dim = self.qkv.weight.shape[0], self.qkv.weight.shape[1]
        arch_weight_matrix = torch.outer(alphas, betas)

        mixed_weight = 0 + weight

        for alpha_idx, in_dim in enumerate(in_feature_dims):
            for beta_idx, n_heads in enumerate(n_head_choices):
                out_dim = (64 * 3 * n_heads)
                in_padding = max_in_dim - in_dim
                out_padding = max_out_dim - out_dim

                out_padding = 0 if out_padding < 0 else out_padding

                sub_weight = self.qkv.weight[:out_dim, :in_dim]
                sub_weight = F.pad(sub_weight, (0, in_padding, 0, out_padding), 'constant', 0)
                mixed_weight += arch_weight_matrix[alpha_idx, beta_idx] * sub_weight

        return mixed_weight

    def _compute_combined_bias(self, bias, n_head_choices, betas):
        if self.qkv.bias is None:
            return None

        max_out_dim = self.qkv.bias.shape[0]
        mixed_bias = 0 + bias

        for beta, n_heads in zip(betas, n_head_choices):
            out_dim = (64 * 3 * n_heads)
            out_padding = max_out_dim - out_dim
            out_padding = 0 if out_padding < 0 else out_padding

            sub_bias = self.qkv.bias[:out_dim]
            sub_bias = F.pad(sub_bias, (0, out_padding), 'constant', 0)

            mixed_bias += beta * sub_bias

        return mixed_bias

    def forward(self, x, alphas, betas):
        mixed_weight = self._compute_combined_weight(self.qkv.weight, self.in_feature_dims, self.n_head_choices, alphas, betas)
        mixed_bias = self._compute_combined_bias(self.qkv.bias, self.n_head_choices, betas)

        out = self.qkv(x, mixed_weight, mixed_bias)

        return out


if __name__ == "__main__":
    qkv = qkv_super(super_in_dim=240, super_out_dim=240*3)
    qkv_lora = WeightEntangledLoraQKV(qkv, in_feature_dims=[24, 192, 240], n_head_choices=[1, 4])

    alphas = F.softmax(torch.randn(3))
    betas = F.softmax(torch.randn(2))
    x = torch.randn(3, 100, 240)

    out = qkv_lora(x, alphas, betas)
