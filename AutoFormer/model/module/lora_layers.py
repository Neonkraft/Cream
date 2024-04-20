#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self,
    ):
        self.r = 0
        self.merged = False

    @abstractmethod
    def _initialize_AB(self) -> None:  # noqa: N802
        pass

    def activate_lora(
        self,
        r: int,
        lora_alpha: int = 1,
        lora_dropout_rate: float = 0,
        merge_weights: bool = True,
    ) -> None:
        assert self.r == 0, "rank can only be changed once"
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout_rate
        self.merge_weights = merge_weights
        if lora_dropout_rate > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout_rate)
        else:
            self.lora_dropout = lambda x: x
        self._initialize_AB()

class LoraLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        linear: nn.Linear,
        **kwargs
    ):
        nn.Linear.__init__(self, linear.in_features, linear.out_features, **kwargs)
        self.weight = linear.weight
        self.bias = linear.bias

        LoRALayer.__init__(self)

        # self.reset_parameters() # we don't want to reset the parameters

    def _initialize_AB(self):
        assert self.r > 0, "Rank should be greater than 0 to use LoRA"

        self.lora_A = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.r)))
        self.scaling = self.lora_alpha / self.r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

    # def reset_parameters(self):
    #     nn.Linear.reset_parameters(self)
    #     if hasattr(self, 'lora_A'):
    #         # initialize B the same way as the default for nn.Linear and A to zero
    #         # this is different than what is described in the paper but should not affect performance
    #         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    #         nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor, weight=None, bias=None):
        w = weight if weight is not None else self.weight
        b = bias if bias is not None else self.bias

        if self.r > 0 and not self.merged:
            result = F.linear(x, w, bias=b)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, w, bias=b)