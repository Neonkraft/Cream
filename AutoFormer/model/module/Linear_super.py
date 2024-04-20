import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.module.lora_layers import LoraLinear

class LinearSuper(nn.Linear):
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
        self._reset_parameters(bias, uniform_, non_linear)
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
    sample_weight = sample_weight[:sample_out_dim, :]

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias


class WeightEntangledLoraLinear(nn.Module):
    def __init__(self, linear, in_feature_dims: list[int], out_feature_dims=None):
        super().__init__()
        self.linear = LoraLinear(linear)
        self.in_feature_dims = in_feature_dims
        self.out_feature_dims = out_feature_dims

    def activate_lora(self, *args, **kwargs):
        self.linear.activate_lora(*args, **kwargs)

    def _compute_combined_weight_in_features(self, weight, in_feature_dims, alphas):
        mixed_weight = 0 + weight

        for in_features_dim, alpha in zip(in_feature_dims, alphas):
            pad_width = weight.shape[1] - in_features_dim
            sub_weight = weight[:, :in_features_dim]
            sub_weight = F.pad(sub_weight, (0, pad_width), 'constant', 0)
            mixed_weight += alpha * sub_weight

        return mixed_weight

    def _compute_combined_weight_in_and_out_features(self, weight, in_feature_dims, out_features, alphas, betas):
        max_out_dim, max_in_dim = self.linear.weight.shape[0], self.linear.weight.shape[1]
        arch_weight_matrix = torch.outer(alphas, betas)

        mixed_weight = 0 + weight

        for alphas_idx, in_features_dim in enumerate(in_feature_dims):
            for betas_idx, out_features_dim in enumerate(out_features):
                pad_in = max_in_dim - in_features_dim
                pad_out = max_out_dim - out_features_dim

                sub_weight = weight[:out_features_dim, :in_features_dim]
                sub_weight = F.pad(sub_weight, (0, pad_in, 0, pad_out), 'constant', 0)
                mixed_weight += arch_weight_matrix[alphas_idx, betas_idx] * sub_weight

        return mixed_weight

    def _compute_combined_bias(self, bias, out_feature_dims, betas):
        if self.linear.bias is None:
            return None

        max_out_dim = self.linear.bias.shape[0]
        mixed_bias = 0 + bias

        for beta, out_features_dim in zip(betas, out_feature_dims):
            pad_out = max_out_dim - out_features_dim
            sub_bias = self.linear.bias[:out_features_dim]
            sub_bias = F.pad(sub_bias, (0, pad_out), 'constant', 0)

            mixed_bias += beta * sub_bias

        return mixed_bias

    def forward(self, x, alphas, betas=None):
        if self.out_feature_dims is None:
            merged_weight = self._compute_combined_weight_in_features(self.linear.weight, self.in_feature_dims, alphas)
            mixed_bias = self.linear.bias
        else:
            assert betas is not None, "Betas must be provided when out_feature_dims is not None"
            merged_weight = self._compute_combined_weight_in_and_out_features(self.linear.weight, self.in_feature_dims, self.out_feature_dims, alphas, betas)
            mixed_bias = self._compute_combined_bias(self.linear.bias, self.out_feature_dims, betas)

        out = self.linear(x, merged_weight, mixed_bias)

        return out
