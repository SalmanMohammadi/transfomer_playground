import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor
from einops import einsum
import math


class LayerNorm(nn.Module):
    """
    LayerNorm is kind of like BatchNorm (which normalizes model weights across the b dimension)
        but works across the model embedding dimension instead of the b dimension.
        It then applies a 'scale and shift' linear operation which intuitively allows
        the model to alter the shape and location of its isotropic weights around.
    Args:
        d_model: model embedding space dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        # These are initialized to one and zero so that the model will begin
        # training with no alterations to the isotropic gaussian weight prior.
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor):
        """
        Applies layer norm to a residual stream vector `x` by normalizing:
            x = (x - x_mu) / x.std()
            x = xw^T + b
        Args:
            x: A residual stream tensor of shape (b, position, d_model)
        """
        # in practice we can divide by the variance + epsilon (1e-7)
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        return self.weight * x + self.bias


class Attention(nn.Module):
    def __init__(self, n_heads: int, d_head: int, d_model: int):
        self.Q_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.xaxier_uniform_(self.Q_w)
        self.Q_b = nn.Parameter(torch.empty(n_heads, d_head))

        self.K_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.xaxier_uniform_(self.K_w)
        self.K_b = nn.Parameter(torch.empty(n_heads, d_head))

        self.V_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.xaxier_uniform_(self.V_w)
        self.V_b = nn.Parameter(torch.empty(n_heads, d_head))

        self.O_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.xaxier_uniform_(self.O_w)
        self.O_b = nn.Parameter(torch.empty(n_heads, d_head))

    def forward(self, x: Tensor):
        """
        Applies a causaul attention mechanism to a residual stream vector `x`.
        Args:
            x: A residual stream tensor of shape (b, position, d_model)
        """
        # the model selects relevant positions by predicting indicies of queries
        queries = einsum("b pos d_m, head d_m d_h -> b pos head d_h", x, self.Q_w) + self.Q_b
        # ... and the corresponding lookup table to the values
        keys = einsum("b pos d_m, head d_m d_h -> b pos head d_h", x, self.K_w) + self.K_b

        # applying the attention operation
        attention_logits = einsum(
            "b q_pos head d_h, b k_pos head d_h -> b head q_pos k_pos", queries, keys
        ) / math.sqrt(self.d_h)

        # masking the upper diagonal - the model shouldn't be able to look ahead
        attention_mask = torch.triu(torch.ones((attention_logits.shape[-2], attention_logits.shape[-1])), diagonal=1)
        attention_masked = attention_logits.masked_fill_(attention_mask.bool().to(attention_logits.device), -1e5)

        # applying softmax so we can find the key with the highest probability
        attention_scores = attention_masked.softmax(-1)

        # the model also has internal feature embeddings for the values at the positions its interested in
        values = einsum("b pos d_m, head d_m d_h -> b pos head d_h", x, self.V_w) + self.V_b
        # and now selects the values based on the keys which are most relevant to the query
        attention = einsum("b head q_pos k_pos, b k_pos head d_h -> b q_pos head d_h", attention, values)

        # a final linear projection helps the model decide which subspace in the residual stream to write to.
        return einsum("b q_pos head d_h, head d_h d_model -> b q_pos d_model", attention, self.O_w) + self.O_b


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, activation=F.gelu):
        super().__init__()
        self.In_w = nn.Parameter(torch.empty((d_model, d_mlp)))
        nn.init.xavier_normal_(self.W_in, std=self.init_range)
        self.b_in = nn.Parameter(torch.empty((d_mlp)))

        self.Out_w = nn.Parameter(torch.empty((d_mlp, d_model)))
        nn.init.xavier_normal_(self.W_out, std=self.init_range)
        self.Out_b = nn.Parameter(torch.empty((d_model)))

        self.activation = activation

    def forward(self, x: Tensor):
        # This blcok just seems to perform some kind of learned nonlinear operation after attention has been applied to the residual stream.
        x = einsum("b pos d_m, d_m d_mlp -> b pos d_mlp", x, self.In_w) + self.In_b
        x = self.activation(x)
        x = einsum("b pos d_mlp, d_mlp d_m -> b pos d_m", x, self.Out_w) + self.Out_b


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, d_mlp, n_heads):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = Attention(n_heads, d_head, d_model)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, resid_pre):
        norm = self.ln1(resid_pre)

        resid_post_attn = resid_pre + self.attn(norm)

        norm_resid_post = self.ln2(resid_post_attn)
        return resid_post_attn + self.mlp(norm_resid_post)
