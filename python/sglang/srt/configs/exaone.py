# coding=utf-8
# Copyright 2024 The LG AI Research EXAONE Lab. All rights reserved.
# Copyright 2024 The LG CNS AI Engineering Team.
# Copyright 2023-2024 SGLang Team.
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
"""EXAONE model configuration"""

from typing import Any, Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

EXAONE_PRETRAINED_CONFIG_ARCHIVE_MAP: Dict[str, Any] = {}


# ruff: noqa: E501
class ExaoneConfig(PretrainedConfig):
    '\n    This is the configuration class to store the configuration of a :class:`~transformers.ExaoneModel`. It is used to\n    instantiate a EXAONE model according to the specified arguments, defining the model architecture. Instantiating a\n    configuration with the defaults will yield a similar configuration to that of the Exaone\n\n    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model\n    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.\n\n\n    Args:\n        vocab_size (:obj:`int`, `optional`, defaults to 102400):\n            Vocabulary size of the EXAONE model. Defines the number of different tokens that can be represented by the\n            :obj:`inputs_ids` passed when calling :class:`~transformers.ExaoneModel`. Vocabulary size of the model.\n            Defines the different tokens that can be represented by the `inputs_ids` passed to the forward method of\n            :class:`~transformers.EXAONEModel`.\n        max_position_embeddings (:obj:`int`, `optional`, defaults to 2048):\n            The maximum sequence length that this model might ever be used with. Typically set this to something large\n            just in case (e.g., 512 or 1024 or 2048).\n        hidden_size (:obj:`int`, `optional`, defaults to 2048):\n            Dimensionality of the encoder layers and the pooler layer.\n        num_layers (:obj:`int`, `optional`, defaults to 32):\n            Number of hidden layers in the Transformer encoder.\n        num_attention_heads (:obj:`int`, `optional`, defaults to 32):\n            Number of attention heads for each attention layer in the Transformer decoder.\n        num_key_value_heads (:obj:`int`, `optional`):\n            This is the number of key_value heads that should be used to implement Grouped Query Attention. If\n            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if\n            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When\n            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed\n            by meanpooling all the original heads within that group. For more details checkout [this\n            paper]([external reference omitted]). If it is not specified, will default to\n            `num_attention_heads`.\n        intermediate_size (:obj:`int`, `optional`, defaults to `hidden_size * 4`):\n            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.\n        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"silu"`):\n            The non-linear activation function (function or string) in the decoder.\n        rope_theta (:obj:`float`, `optional`, defaults to 10000.0):\n            The base period of the RoPE embeddings.\n        rope_scaling (:obj:`Dict`, `optional`):\n            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type\n            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value\n            accordingly.\n            Expected contents:\n                `rope_type` (:obj:`str`):\n                    The sub-variant of RoPE to use. Can be one of [\'default\', \'linear\', \'dynamic\', \'yarn\', \'longrope\',\n                    \'llama3\'], with \'default\' being the original RoPE implementation.\n                `factor` (:obj:`float`, `optional`):\n                    Used with all rope types except \'default\'. The scaling factor to apply to the RoPE embeddings. In\n                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *\n                    original maximum pre-trained length.\n                `original_max_position_embeddings` (:obj:`int`, `optional`):\n                    Used with \'dynamic\', \'longrope\' and \'llama3\'. The original max position embeddings used during\n                    pretraining.\n                `attention_factor` (:obj:`float`, `optional`):\n                    Used with \'yarn\' and \'longrope\'. The scaling factor to be applied on the attention\n                    computation. If unspecified, it defaults to value recommended by the implementation, using the\n                    `factor` field to infer the suggested value.\n                `beta_fast` (:obj:`float`, `optional`):\n                    Only used with \'yarn\'. Parameter to set the boundary for extrapolation (only) in the linear\n                    ramp function. If unspecified, it defaults to 32.\n                `beta_slow` (:obj:`float`, `optional`):\n                    Only used with \'yarn\'. Parameter to set the boundary for interpolation (only) in the linear\n                    ramp function. If unspecified, it defaults to 1.\n                `short_factor` (:obj:`List[float]`, `optional`):\n                    Only used with \'longrope\'. The scaling factor to be applied to short contexts (<\n                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden\n                    size divided by the number of attention heads divided by 2\n                `long_factor` (:obj:`List[float]`, `optional`):\n                    Only used with \'longrope\'. The scaling factor to be applied to long contexts (<\n                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden\n                    size divided by the number of attention heads divided by 2\n                `low_freq_factor` (:obj:`float`, `optional`):\n                    Only used with \'llama3\'. Scaling factor applied to low frequency components of the RoPE\n                `high_freq_factor` (:obj:`float`, `optional`):\n                    Only used with \'llama3\'. Scaling factor applied to high frequency components of the RoPE\n        embed_dropout (:obj:`float`, `optional`, defaults to 0.0):\n            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.\n        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):\n            The dropout ratio for the attention probabilities.\n        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):\n            The epsilon used by the layer normalization layers.\n        initializer_range (:obj:`float`, `optional`, defaults to 0.02):\n            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.\n        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):\n            Whether or not the model should return the last key/values attentions (not used by all models). Only\n            relevant if ``configs.is_decoder=True``.\n        bos_token_id (:obj:`int`, `optional`, defaults to 0):\n            Beginning of stream token id.\n        eos_token_id (:obj:`int`, `optional`, defaults to 2):\n            End of stream token id.\n        tie_word_embeddings (:obj:`bool`, `optional`, defaults to :obj:`True`):\n            Whether to tie weight embeddings\n        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            If True, use gradient checkpointing to save memory at the expense of slower backward pass.\n\n        Example::\n\n            >>> from transformers import EXAONEModel, ExaoneConfig\n\n            >>> # Initializing a EXAONE configuration\n            >>> configuration = ExaoneConfig()\n\n            >>> # Initializing a model from configuration\n            >>> model = EXAONEModel(configuration)\n\n            >>> # Accessing the model configuration\n            >>> configuration = model.configs\n    '

    model_type = "exaone"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=102400,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        intermediate_size=None,
        activation_function="silu",
        rope_theta=10000.0,
        rope_scaling=None,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if intermediate_size:
            self.intermediate_size = intermediate_size
        else:
            self.intermediate_size = hidden_size * 4
        self.activation_function = activation_function
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
