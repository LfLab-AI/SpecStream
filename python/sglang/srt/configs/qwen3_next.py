# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3Hybrid model configuration"""

import enum

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.configs.update_config import adjust_tp_num_heads_if_necessary
from sglang.srt.utils import is_cpu

logger = logging.get_logger(__name__)
_is_cpu = is_cpu()


class HybridLayerType(enum.Enum):
    full_attention = "attention"
    linear_attention = "linear_attention"


class Qwen3NextConfig(PretrainedConfig):
    '\n    This is the configuration class to store the configuration of a [`Qwen3NextModel`]. It is used to instantiate a\n    Qwen3-Next model according to the specified arguments, defining the model architecture.\n    Instantiating a configuration with the defaults will yield a similar configuration to that of\n    Qwen3-Next-80B-A3B-Instruct [Qwen/Qwen3-Next-80B-A3B-Instruct]([external reference omitted]).\n\n    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the\n    documentation from [`PretrainedConfig`] for more information.\n\n\n    Args:\n        vocab_size (`int`, *optional*, defaults to 151936):\n            Vocabulary size of the model. Defines the number of different tokens that can be represented by the\n            `inputs_ids`.\n        hidden_size (`int`, *optional*, defaults to 2048):\n            Dimension of the hidden representations.\n        intermediate_size (`int`, *optional*, defaults to 5632):\n            Dimension of the MLP representations.\n        num_hidden_layers (`int`, *optional*, defaults to 48):\n            Number of hidden layers in the Transformer encoder.\n        num_attention_heads (`int`, *optional*, defaults to 16):\n            Number of attention heads for each attention layer in the Transformer encoder.\n        num_key_value_heads (`int`, *optional*, defaults to 2):\n            This is the number of key_value heads that should be used to implement Grouped Query Attention. If\n            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if\n            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When\n            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed\n            by meanpooling all the original heads within that group. For more details checkout [this\n            paper]([external reference omitted]). If it is not specified, will default to `32`.\n        hidden_act (`str`, *optional*, defaults to `"silu"`):\n            The non-linear activation function in the decoder.\n        max_position_embeddings (`int`, *optional*, defaults to 32768):\n            The maximum sequence length that this model might ever be used with.\n        initializer_range (`float`, *optional*, defaults to 0.02):\n            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.\n        rms_norm_eps (`float`, *optional*, defaults to 1e-06):\n            The epsilon used by the rms normalization layers.\n        use_cache (`bool`, *optional*, defaults to `True`):\n            Whether or not the model should return the last key/values attentions (not used by all models). Only\n            relevant if `config.is_decoder=True`.\n        tie_word_embeddings (`bool`, *optional*, defaults to `False`):\n            Whether the model\'s input and output word embeddings should be tied.\n        rope_theta (`float`, *optional*, defaults to 10000.0):\n            The base period of the RoPE embeddings.\n        rope_scaling (`Dict`, *optional*):\n            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type\n            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value\n            accordingly.\n            Expected contents:\n                `rope_type` (`str`):\n                    The sub-variant of RoPE to use. Can be one of [\'default\', \'linear\', \'dynamic\', \'yarn\', \'longrope\',\n                    \'llama3\'], with \'default\' being the original RoPE implementation.\n                `factor` (`float`, *optional*):\n                    Used with all rope types except \'default\'. The scaling factor to apply to the RoPE embeddings. In\n                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *\n                    original maximum pre-trained length.\n                `original_max_position_embeddings` (`int`, *optional*):\n                    Used with \'dynamic\', \'longrope\' and \'llama3\'. The original max position embeddings used during\n                    pretraining.\n                `attention_factor` (`float`, *optional*):\n                    Used with \'yarn\' and \'longrope\'. The scaling factor to be applied on the attention\n                    computation. If unspecified, it defaults to value recommended by the implementation, using the\n                    `factor` field to infer the suggested value.\n                `beta_fast` (`float`, *optional*):\n                    Only used with \'yarn\'. Parameter to set the boundary for extrapolation (only) in the linear\n                    ramp function. If unspecified, it defaults to 32.\n                `beta_slow` (`float`, *optional*):\n                    Only used with \'yarn\'. Parameter to set the boundary for interpolation (only) in the linear\n                    ramp function. If unspecified, it defaults to 1.\n                `short_factor` (`List[float]`, *optional*):\n                    Only used with \'longrope\'. The scaling factor to be applied to short contexts (<\n                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden\n                    size divided by the number of attention heads divided by 2\n                `long_factor` (`List[float]`, *optional*):\n                    Only used with \'longrope\'. The scaling factor to be applied to long contexts (<\n                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden\n                    size divided by the number of attention heads divided by 2\n                `low_freq_factor` (`float`, *optional*):\n                    Only used with \'llama3\'. Scaling factor applied to low frequency components of the RoPE\n                `high_freq_factor` (`float`, *optional*):\n                    Only used with \'llama3\'. Scaling factor applied to high frequency components of the RoPE\n        partial_rotary_factor (`float`, *optional*, defaults to 0.25):\n            Percentage of the query and keys which will have rotary embedding.\n        attention_bias (`bool`, *optional*, defaults to `False`):\n            Whether to use a bias in the query, key, value and output projection layers during self-attention.\n        attention_dropout (`float`, *optional*, defaults to 0.0):\n            The dropout ratio for the attention probabilities.\n        head_dim (`int`, *optional*, defaults to 256):\n            Projection weights dimension in multi-head attention.\n        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):\n            Kernel size of the convolution used in linear attention layers.\n        linear_key_head_dim (`int`, *optional*, defaults to 128):\n            Dimension of each key head in linear attention.\n        linear_value_head_dim (`int`, *optional*, defaults to 128):\n            Dimension of each value head in linear attention.\n        linear_num_key_heads (`int`, *optional*, defaults to 16):\n            Number of key heads used in linear attention layers.\n        linear_num_value_heads (`int`, *optional*, defaults to 32):\n            Number of value heads used in linear attention layers.\n        decoder_sparse_step (`int`, *optional*, defaults to 1):\n            The frequency of the MoE layer.\n        moe_intermediate_size (`int`, *optional*, defaults to 512):\n            Intermediate size of the routed expert.\n        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):\n            Intermediate size of the shared expert.\n        num_experts_per_tok (`int`, *optional*, defaults to 10):\n            Number of selected experts.\n        num_experts (`int`, *optional*, defaults to 512):\n            Number of routed experts.\n        norm_topk_prob (`bool`, *optional*, defaults to `True`):\n            Whether to normalize the topk probabilities.\n        output_router_logits (`bool`, *optional*, defaults to `False`):\n            Whether or not the router logits should be returned by the model. Enabling this will also\n            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.\n        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):\n            The aux loss factor for the total loss.\n        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):\n            Indicate which layers use Qwen3NextMLP rather than Qwen3NextSparseMoeBlock\n            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers\n            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.\n        layer_types (`list[str]`, *optional*, defaults to None):\n            Types of each layer (attention or linear).\n\n    ```python\n    >>> from transformers import Qwen3NextModel, Qwen3NextConfig\n\n    >>> # Initializing a Qwen3Next style configuration\n    >>> configuration =  Qwen3NextConfig()\n\n    >>> # Initializing a model from the Qwen3-Next-80B-A3B style configuration\n    >>> model = Qwen3NextModel(configuration)\n\n    >>> # Accessing the model configuration\n    >>> configuration = model.config\n    ```\n    '

    model_type = "qwen3_next"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=10,
        num_experts=512,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=[],
        layer_types=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        # linear attention (gdn now part)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers

    @property
    def layers_block_type(self):
        layer_type_list = []

        for l in range(self.num_hidden_layers):
            if (l + 1) % self.full_attention_interval == 0:
                layer_type_list.append(HybridLayerType.full_attention.value)
            else:
                layer_type_list.append(HybridLayerType.linear_attention.value)

        return layer_type_list

    @property
    def linear_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.linear_attention.value
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.full_attention.value
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        if _is_cpu:
            world_size = get_attention_tp_size()
            adjust_tp_num_heads_if_necessary(self, world_size, False)

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.linear_value_head_dim * self.linear_num_value_heads,
            n_groups=self.linear_num_key_heads,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
        )

        return Mamba2CacheParams(
            shape=shape, layers=self.linear_layer_ids, dtype=mamba2_state_dtype(self)
        )
