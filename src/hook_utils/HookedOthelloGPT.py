from __future__ import annotations

import torch
import torch.nn as nn


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = None

    def forward(self, x):
        return x


def is_othello_gpt(model: nn.Module) -> bool:
    """Duck-typed check for the local Othello GPT architecture."""
    return all(
        hasattr(model, attribute)
        for attribute in ("tok_emb", "pos_emb", "blocks", "ln_f", "head", "block_size")
    ) and isinstance(getattr(model, "blocks"), nn.Sequential)


def _is_othello_block(module: nn.Module) -> bool:
    return all(hasattr(module, attribute) for attribute in ("ln1", "ln2", "attn", "mlp"))


def _is_othello_attention(module: nn.Module) -> bool:
    return all(
        hasattr(module, attribute)
        for attribute in (
            "key",
            "query",
            "value",
            "proj",
            "attn_drop",
            "resid_drop",
            "n_head",
            "mask",
        )
    )


def _is_othello_mlp(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Sequential)
        and len(module) == 4
        and isinstance(module[0], nn.Linear)
        and isinstance(module[1], nn.GELU)
        and isinstance(module[2], nn.Linear)
        and isinstance(module[3], nn.Dropout)
    )


def hooked_forward_attention(self, x, layer_past=None, only_last=-1):
    del layer_past

    batch_size, seq_len, n_embd = x.size()
    head_dim = n_embd // self.n_head

    key_states = self.key(x).view(batch_size, seq_len, self.n_head, head_dim).transpose(
        1, 2
    )
    query_states = self.query(x).view(
        batch_size, seq_len, self.n_head, head_dim
    ).transpose(1, 2)
    value_states = self.value(x).view(
        batch_size, seq_len, self.n_head, head_dim
    ).transpose(1, 2)

    key_states = self.hook_key_states(key_states)
    query_states = self.hook_query_states(query_states)

    attn_weights = (query_states @ key_states.transpose(-2, -1)) * (head_dim**-0.5)
    attn_weights = attn_weights.masked_fill(
        self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
    )
    if only_last != -1:
        attn_weights[:, :, -only_last:, :-only_last] = float("-inf")
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = self.attn_drop(attn_weights)
    attn_weights = self.hook_attn_pattern(attn_weights)

    value_states = self.hook_value_states_post_attn(value_states)
    attn_output = attn_weights @ value_states
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
    attn_output = self.proj(attn_output)
    attn_output = self.hook_o_proj(attn_output)
    attn_output = self.resid_drop(attn_output)
    attn_output = self.hook_attn_out(attn_output)
    return attn_output, attn_weights


def hooked_forward_mlp(self, x):
    hidden = self[0](x)
    hidden = self[1](hidden)
    hidden = self.hook_mlp_mid(hidden)
    hidden = self[2](hidden)
    hidden = self[3](hidden)
    hidden = self.hook_mlp_out(hidden)
    return hidden


def hooked_forward_block(self, x, return_att=False, only_last=-1):
    x = self.hook_resid_pre(x)
    attn_update, attn_pattern = self.attn(self.ln1(x), only_last=only_last)
    x = x + attn_update
    x = self.hook_resid_mid(x)
    x = x + self.mlp(self.ln2(x))
    x = self.hook_resid_post(x)
    if return_att:
        return x, attn_pattern
    return x


def _convert_to_hooked_model(module: nn.Module) -> None:
    for child in module.children():
        if _is_othello_attention(child):
            child.forward = hooked_forward_attention.__get__(child, child.__class__)

        if _is_othello_mlp(child):
            child.forward = hooked_forward_mlp.__get__(child, child.__class__)

        if _is_othello_block(child):
            child.forward = hooked_forward_block.__get__(child, child.__class__)

        _convert_to_hooked_model(child)


def convert_to_hooked_model_othello_gpt(model: nn.Module) -> None:
    """Inject hook points into the local Othello GPT architecture."""
    for layer in model.blocks:
        layer.hook_resid_pre = HookPoint()
        layer.hook_resid_mid = HookPoint()
        layer.hook_resid_post = HookPoint()

        layer.attn.hook_key_states = HookPoint()
        layer.attn.hook_query_states = HookPoint()
        layer.attn.hook_attn_pattern = HookPoint()
        layer.attn.hook_value_states_post_attn = HookPoint()
        layer.attn.hook_o_proj = HookPoint()
        layer.attn.hook_attn_out = HookPoint()

        layer.mlp.hook_mlp_mid = HookPoint()
        layer.mlp.hook_mlp_out = HookPoint()

    _convert_to_hooked_model(model)
