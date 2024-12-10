#!/usr/bin/env python
# encoding: utf-8

from typing import Optional, Tuple

import torch
from torch import nn

import transformers
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb_func
from flash_attn.ops.rms_norm import rms_norm
from flash_attn.ops.fused_dense import FusedDense
from flash_attn.ops.activations import swiglu
from flash_attn.losses.cross_entropy import CrossEntropyLoss


def flash_rms_norm(self, x):
    return rms_norm(x, self.weight, self.variance_epsilon)


class FusedLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = FusedDense(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = FusedDense(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = FusedDense(self.intermediate_size, self.hidden_size, bias=False)

        # self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
        return down_proj


class FlashRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, base=10000.0, max_position_embeddings=2048, interleaved=False, scale_base=None,
                 scaling_factor=1.0, device=None):
        """
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
            scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                 / (1.4 * dim) if scale_base is not None else None)
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        print("use rotary patch: --init--")

    def _compute_inv_freq(self, device=None):
        return 1 / (self.base ** (torch.arange(0, self.dim, 2, device=device,
                                                 dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (seqlen > self._seq_len_cached or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            t /= self.scaling_factor
            # We want fp32 here as well since inv_freq will be multiplied with t, and the output
            # will be large. Having it in bf16 will lose a lot of precision and cause the
            # cos & sin output to change significantly.
            # We want to recompute self.inv_freq if it was not loaded in fp32
            if self.inv_freq.dtype != torch.float32:
                inv_freq = self.inv_freq.to(torch.float32)
            else:
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = ((torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                          - seqlen // 2) / self.scale_base)
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)
        if self.scale is None:
            return apply_rotary_emb_func(
                q, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                self.interleaved, True # inplace=True
            ), apply_rotary_emb_func(
                k, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                self.interleaved, True # inplace=True
            )
        else:
            assert False


def flash_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim)

    kv_seq_len = key_states.shape[-2]
    past_len = 0
    if past_key_value is not None:
        past_len = past_key_value[0].shape[-2]
        kv_seq_len += past_len
    query_states, key_states = self.rotary_emb(query_states, key_states, past_len)

    attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True
        )

    # bsz, nh, len, hd
    output = attn_output.reshape(bsz, q_len, self.hidden_size)
    return self.o_proj(output),  None, None



def replace_llama_attn_with_sdpa():
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = flash_rms_norm
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = FlashRotaryEmbedding
    transformers.models.llama.modeling_llama.CrossEntropyLoss = CrossEntropyLoss
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flash_forward
