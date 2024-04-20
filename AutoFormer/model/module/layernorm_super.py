import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim


class WeightEntangledLayerNorm(nn.Module):
    def __init__(self, layernorm: nn.LayerNorm, in_feature_dims: list[int]):
        super().__init__()
        self.layernorm = layernorm
        self.in_feature_dims = in_feature_dims

    def _combine_weights(self, weight, in_feature_dims, alphas):
        max_dim = weight.shape[0]
        mixed_weight = 0 + weight

        for in_dim, alpha in zip(in_feature_dims, alphas):
            pad_width = max_dim - in_dim
            sub_weight = weight[:in_dim]
            sub_weight = F.pad(sub_weight, (0, pad_width), 'constant', 0)
            mixed_weight += alpha * sub_weight

        return mixed_weight

    def _compute_combined_weight(self, weight, in_feature_dims, alphas):
        return self._combine_weights(weight, in_feature_dims, alphas)

    def _compute_combined_bias(self, bias, in_feature_dims, alphas):
        if bias is None:
            return None

        return self._combine_weights(bias, in_feature_dims, alphas)

    def forward(self, x, alphas):
        mixed_weight = self._compute_combined_weight(self.layernorm.weight, self.in_feature_dims, alphas)
        mixed_bias = self._compute_combined_bias(self.layernorm.bias, self.in_feature_dims, alphas)
        out = F.layer_norm(x, (self.layernorm.weight.shape[0],), weight=mixed_weight, bias=mixed_bias, eps=self.layernorm.eps)

        return out
