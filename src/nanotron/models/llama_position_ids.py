# llama.py -- rewritten to use position_ids everywhere (Qwen2-style)
# input_ids [B, S], position_ids [B, S]
#     │
#     ▼
# Embedding: input_embeds [B*S, H], position_ids [B, S]
#     │
#     ▼
# cu_seqlens [B+1] (from position_ids)
#     │
#     ▼
# for each Decoder Layer:
#     ├─ input_layernorm: [B*S, H]
#     ├─ QKV proj: q, k, v [B*S, n_local_heads, hd]
#     ├─ Rotary: q, k [B*S, n_local_heads, hd]
#     ├─ FlashAttention: attn_output [B*S, n_local_heads, hd]
#     ├─ Output proj: [B*S, H]
#     ├─ Residual add: [B*S, H]
#     ├─ post_attention_layernorm: [B*S, H]
#     ├─ MLP: [B*S, H]
#     └─ Residual add: [B*S, H]
#     │
#     ▼
# final_layer_norm: [B*S, H]
#     │
#     ▼
# lm_head: sharded_logits [B*S, vocab_size//TP]
#     │
#     ▼
# (loss if training)


from typing import Dict, List, Optional, Union

import torch
from flash_attn import bert_padding
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, LlamaConfig, ParallelismArgs
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.utils import checkpoint_method

logger = logging.get_logger(__name__)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        self.freqs_cis: torch.Tensor
        self._initialized_buffer = False

    def init_rotary_embeddings(self):
        if self._initialized_buffer:
            return
        self.register_buffer(
            "freqs_cis",
            torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device="cuda"),
            persistent=False,
        )
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu")[: (self.dim // 2)] / self.dim)
        ).to("cuda")
        t = torch.arange(self.end, device="cuda")
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = torch.view_as_real(complex_freqs)
        self.freqs_cis.copy_(freqs)
        self._initialized_buffer = True

    def forward(
        self,
        position_ids: torch.LongTensor,  # [batch_size*seq_length]
        seq_length: int,
    ):
        # position_ids: [batch_size*seq_length]
        max_pos = position_ids.max().item() if position_ids.numel() > 0 else 0
        while max_pos >= self.end or seq_length >= self.end:
            self.end *= 2
            self._initialized_buffer = False
        if not self._initialized_buffer:
            self.init_rotary_embeddings()
        return self.freqs_cis[position_ids]  # [batch_size*seq_length, dim//2, 2]

    def apply_rotary_pos_emb(self, x, rotary_pos_emb, seq_length):
        # x: [b*s, num_heads, head_dim], rotary_pos_emb: [b*s, dim//2, 2]
        bsh, n_heads, head_dim = x.shape
        x = x.view(bsh, n_heads, head_dim // 2, 2)
        if x.dtype == torch.bfloat16:
            x = x.float()
        complex_x = torch.view_as_complex(x)
        complex_freqs = torch.view_as_complex(rotary_pos_emb)
        x_out = torch.view_as_real(complex_x * complex_freqs.unsqueeze(1)).view(bsh, n_heads, head_dim)
        return x_out.type(x.dtype)


class GLUActivation(nn.Module):
    def __init__(self, act_fn_name: str):
        super().__init__()
        self.act = ACT2FN[act_fn_name]

    def forward(self, merged_states: torch.Tensor):
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        return self.act(gate_states) * up_states


class MLP(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )
        gate_up_contiguous_chunks = (
            config.intermediate_size,
            config.intermediate_size,
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.split_silu_mul = GLUActivation(config.hidden_act)

    def forward(self, hidden_states):
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}


class CoreAttention(nn.Module):
    def __init__(self, config: LlamaConfig, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.is_using_mup = config.is_using_mup
        self.checkpoint_attention = False

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,  # [b*s, n_local_q_heads, d_qk]
        key_states: torch.Tensor,    # [b*s, n_local_kv_heads, d_qk]
        value_states: torch.Tensor,  # [b*s, n_local_kv_heads, d_v]
        position_ids: torch.Tensor,  # [b*s]
        seq_length: int,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        # cu_seqlens: [batch+1]
        if cu_seqlens is None:
            # Compute cu_seqlens from position_ids
            start_indices = torch.where(position_ids.view(-1) == 0)[0]
            cu_seqlens = torch.cat(
                [start_indices, torch.tensor([position_ids.numel()], dtype=torch.int32, device=start_indices.device)]
            ).to(dtype=position_ids.dtype)
        max_seqlen = seq_length
        softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            return_attn_probs=False,
        )
        return attn_output


class CausalSelfAttention(nn.Module, AttachableStore):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        super().__init__()
        assert config.num_attention_heads % tp_pg.size() == 0
        try:
            assert config.num_key_value_heads % tp_pg.size() == 0
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            config.num_key_value_heads = config.num_attention_heads
        assert config.num_attention_heads % config.num_key_value_heads == 0
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        qkv_contiguous_chunks = (
            config.num_attention_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk,
        )
        self.qkv_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.rotary_embedding = RotaryEmbedding(
            dim=self.d_qk,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )
        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )
        self.prefill_kv_len = config.max_position_embeddings

    def forward(
        self,
        hidden_states,  # [batch_size*seq_length, hidden_size]
        position_ids,   # [batch_size, seq_length] where -1 is padding
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        seq_length = position_ids.shape[1]
        position_ids = position_ids.view(-1)  # [batch_size*seq_length]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.n_local_q_heads * self.d_qk, self.n_local_kv_heads * self.d_qk, self.n_local_kv_heads * self.d_qk], dim=-1
        )
        q = q.view(-1, self.n_local_q_heads, self.d_qk)
        k = k.view(-1, self.n_local_kv_heads, self.d_qk)
        v = v.view(-1, self.n_local_kv_heads, self.d_qk)
        rotary_pos_emb = self.rotary_embedding(position_ids=position_ids, seq_length=seq_length)
        q = self.rotary_embedding.apply_rotary_pos_emb(q, rotary_pos_emb, seq_length=seq_length)
        k = self.rotary_embedding.apply_rotary_pos_emb(k, rotary_pos_emb, seq_length=seq_length)
        attn_output = self.attention(
            q, k, v, position_ids=position_ids, seq_length=seq_length, cu_seqlens=cu_seqlens
        )
        output = self.o_proj(attn_output)
        return {"hidden_states": output, "position_ids": position_ids.view(-1, seq_length)}


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)
        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        cu_seqlens: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output = self.attn(hidden_states=hidden_states, position_ids=position_ids, cu_seqlens=cu_seqlens)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual
        return hidden_states, output["position_ids"], cu_seqlens

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> List[torch.Tensor]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, position_ids, cu_seqlens)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        cu_seqlens: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, position_ids, cu_seqlens = self._checkpointed_forward(hidden_states, position_ids, cu_seqlens)
        else:
            hidden_states, position_ids, cu_seqlens = self._core_forward(hidden_states, position_ids, cu_seqlens)
        return {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
        }


class Embedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: LlamaConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):  # [batch_size, seq_length]
        input_ids = input_ids.view(-1)
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds, "position_ids": position_ids}


class LlamaModel(nn.Module):
    """Build pipeline graph"""

    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )
        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "position_ids"},
            module_output_keys={"input_embeds", "position_ids"},
        )
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LlamaDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "position_ids", "cu_seqlens"},
                    module_output_keys={"hidden_states", "position_ids", "cu_seqlens"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )
        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )
        self.cast_to_fp32 = PipelineBlock(
            p2p=self.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        return self.forward_with_hidden_states(input_ids=input_ids, position_ids=position_ids)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
    ):
        output = self.token_position_embeddings(input_ids=input_ids, position_ids=position_ids)
        # Compute cu_seqlens
        if position_ids.numel() > 0:
            start_indices = torch.where(position_ids.view(-1) == 0)[0]
            cu_seqlens = torch.cat(
                [start_indices, torch.tensor([position_ids.numel()], dtype=torch.int32, device=start_indices.device)]
            ).to(dtype=position_ids.dtype)
        else:
            cu_seqlens = None
        decoder_states = {
            "hidden_states": output["input_embeds"],
            "position_ids": output["position_ids"],
            "cu_seqlens": cu_seqlens,
        }
        for decoder_layer in self.decoder:
            decoder_states = decoder_layer(**decoder_states)
        hidden_states = self.final_layer_norm(input=decoder_states["hidden_states"])["hidden_states"]
        sharded_logits = self.lm_head(x=hidden_states)["logits"]
        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]
        return fp32_sharded_logits, hidden_states

    def get_block_compute_costs(self):
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        world_size = self.parallel_context.world_pg.size()
        try:
            num_key_values_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = self.config.num_attention_heads
        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )
        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


class Loss(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup):
        super().__init__()
        self.tp_pg = tp_pg

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [batch_size*seq_length, logits]
        label_ids: torch.Tensor,       # [batch_size, seq_length]
        label_mask: torch.Tensor,      # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        sharded_logits = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)
        loss = sharded_cross_entropy(sharded_logits, label_ids.contiguous(), group=self.tp_pg, dtype=torch.float)
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        return {"loss": loss}


class LossWithZLoss(Loss):
    def __init__(self, tp_pg: dist.ProcessGroup, z_loss_coefficient: float):
        super().__init__(tp_pg)
        self.z_loss_coef = z_loss_coefficient

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [batch_size*seq_length, logits]
        label_ids: torch.Tensor,       # [batch_size, seq_length]
        label_mask: torch.Tensor,      # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        sharded_logits = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)
        loss, z_loss = sharded_cross_entropy(
            sharded_logits, label_ids.contiguous(), group=self.tp_pg, dtype=torch.float, z_loss_coef=self.z_loss_coef
        )
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        z_loss = masked_mean(z_loss.detach(), label_mask, dtype=torch.float)
        return {"loss": loss, "z_loss": z_loss}


class LlamaForTraining(NanotronModel):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        loss_kwargs = {
            "tp_pg": parallel_context.tp_pg,
        }
        if config.z_loss_enabled:
            loss_kwargs["z_loss_coefficient"] = config.z_loss_coefficient
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=LossWithZLoss if config.z_loss_enabled else Loss,
            module_kwargs=loss_kwargs,
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss", "z_loss"} if config.z_loss_enabled else {"loss"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
        )
        if self.config.z_loss_enabled:
            return {"loss": loss["loss"], "z_loss": loss["z_loss"]}
        else:
            return {"loss": loss["loss"]}

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")
        parametrizator = parametrizator_cls(config=config)
        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        model = self
        initialized_parameters = set()
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        module_id_to_prefix[id(model)] = ""
        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)
            module_name, param_name = param_name.rsplit(".", 1)
            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"
            if full_param_name in initialized_parameters:
                continue
            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)
            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)
        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_embeddings_lm_head_tied_names(self):
        if self.config.tie_word_embeddings is True:
            return ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        else:
            return []

    def get_block_compute_costs(self):
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
):
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * num_heads * hidden_size_per_head
        + 2 * num_layers * batch_size * seq_len * (hidden_size) * 2 * num_key_value_heads * hidden_size_per_head
    )
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * seq_len
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * hidden_size_per_head
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * hidden_size
    )
    decoder_ffn_1_flops_fwd = 4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    decoder_ffn_2_flops_fwd = 2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size
    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size
    model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)
    hardware_flops = model_flops
    return model_flops, hardware_flops
