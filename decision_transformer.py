# decision_transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


NormType = Literal["pre", "post"]


@dataclass
class DecisionTransformerConfig:
    state_dim: int
    act_dim: int

    hidden_size: int = 128
    n_layer: int = 3
    n_head: int = 1
    context_len: int = 20

    max_timestep: int = 1000
    dropout: float = 0.1

    norm_type: NormType = "pre"
    final_layer_norm: bool = True
    action_tanh: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_head: int, dropout: float):
        super().__init__()

        if hidden_size % n_head != 0:
            raise ValueError("hidden_size must be divisible by n_head")

        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask.view(1, 1, T, T), self._mask_value(scores))

        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool).view(B, 1, 1, T)
            scores = scores.masked_fill(~key_mask, self._mask_value(scores))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_drop(self.out_proj(y))

        if attention_mask is not None:
            y = y * attention_mask.to(y.dtype).unsqueeze(-1)

        return y

    @staticmethod
    def _mask_value(scores):
        if scores.dtype in (torch.float16, torch.bfloat16):
            return -1e4
        return -1e9


class MLP(nn.Module):
    def __init__(self, hidden_size, dropout, expansion=4):
        super().__init__()
        inner_dim = expansion * hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_head, dropout, norm_type="pre"):
        super().__init__()
        if norm_type not in ("pre", "post"):
            raise ValueError("norm_type must be 'pre' or 'post'")
        self.norm_type = norm_type
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_head, dropout)
        self.mlp = MLP(hidden_size, dropout)

    def forward(self, x, attention_mask=None):
        if self.norm_type == "pre":
            x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
            x = x + self.mlp(self.ln_2(x))
            return x
        x = self.ln_1(x + self.attn(x, attention_mask=attention_mask))
        x = self.ln_2(x + self.mlp(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        self.config = config

        D = config.hidden_size

        self.embed_timestep = nn.Embedding(config.max_timestep + 1, D)
        self.embed_return = nn.Linear(1, D)
        self.embed_state = nn.Linear(config.state_dim, D)
        self.embed_action = nn.Linear(config.act_dim, D)

        self.embed_ln = nn.LayerNorm(D)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=D,
                n_head=config.n_head,
                dropout=config.dropout,
                norm_type=config.norm_type,
            )
            for _ in range(config.n_layer)
        ])

        self.final_ln = nn.LayerNorm(D) if config.final_layer_norm else nn.Identity()

        self.predict_state = nn.Linear(D, config.state_dim)
        self.predict_return = nn.Linear(D, 1)

        if config.action_tanh:
            self.predict_action = nn.Sequential(
                nn.Linear(D, config.act_dim),
                nn.Tanh(),
            )
        else:
            self.predict_action = nn.Linear(D, config.act_dim)

        self.apply(self._init_weights)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        B, K, _ = states.shape

        if K > self.config.context_len:
            raise ValueError(
                f"Got sequence length K={K}, but config.context_len={self.config.context_len}"
            )

        if returns_to_go.ndim == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        timesteps = timesteps.clamp(min=0, max=self.config.max_timestep).long()

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        return_embeddings = self.embed_return(returns_to_go) + time_embeddings

        token_embeddings = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings),
            dim=2,
        )
        x = token_embeddings.reshape(B, 3 * K, self.config.hidden_size)

        x = self.embed_ln(x)
        x = self.drop(x)

        if attention_mask is None:
            token_attention_mask = torch.ones(B, 3 * K, dtype=torch.bool, device=states.device)
        else:
            token_attention_mask = attention_mask.to(torch.bool)
            token_attention_mask = token_attention_mask.unsqueeze(-1).expand(B, K, 3)
            token_attention_mask = token_attention_mask.reshape(B, 3 * K)

        for block in self.blocks:
            x = block(x, attention_mask=token_attention_mask)

        x = self.final_ln(x)
        x = x.reshape(B, K, 3, self.config.hidden_size)

        action_preds = self.predict_action(x[:, :, 1])
        state_preds = self.predict_state(x[:, :, 2])
        return_preds = self.predict_return(x[:, :, 2])

        return {
            "action_preds": action_preds,
            "state_preds": state_preds,
            "return_preds": return_preds,
        }

    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps, state_mean=None, state_std=None):
        self.eval()
        device = next(self.parameters()).device
        K = self.config.context_len

        states = states.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.float32)
        returns_to_go = returns_to_go.to(device=device, dtype=torch.float32)
        timesteps = timesteps.to(device=device, dtype=torch.long)

        if returns_to_go.ndim == 1:
            returns_to_go = returns_to_go.unsqueeze(-1)

        states = states[-K:]
        actions = actions[-K:]
        returns_to_go = returns_to_go[-K:]
        timesteps = timesteps[-K:]

        if state_mean is not None and state_std is not None:
            state_mean = state_mean.to(device=device, dtype=torch.float32)
            state_std = state_std.to(device=device, dtype=torch.float32)
            states = (states - state_mean) / state_std

        T = states.shape[0]
        pad_len = K - T

        if pad_len > 0:
            states = torch.cat([torch.zeros(pad_len, self.config.state_dim, device=device), states], dim=0)
            actions = torch.cat([torch.zeros(pad_len, self.config.act_dim, device=device), actions], dim=0)
            returns_to_go = torch.cat([torch.zeros(pad_len, 1, device=device), returns_to_go], dim=0)
            timesteps = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device), timesteps], dim=0)

        attention_mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.bool, device=device),
            torch.ones(T, dtype=torch.bool, device=device),
        ], dim=0)

        out = self.forward(
            states=states.unsqueeze(0),
            actions=actions.unsqueeze(0),
            returns_to_go=returns_to_go.unsqueeze(0),
            timesteps=timesteps.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )

        return out["action_preds"][0, -1]

    @staticmethod
    def masked_action_mse(action_preds, action_targets, attention_mask=None):
        per_step_loss = (action_preds - action_targets).pow(2).mean(dim=-1)
        if attention_mask is None:
            return per_step_loss.mean()
        mask = attention_mask.to(per_step_loss.dtype)
        return (per_step_loss * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
