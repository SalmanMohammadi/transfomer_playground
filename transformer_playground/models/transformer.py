import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import einsum, repeat, rearrange
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies layer norm to a residual stream vector `x` by normalizing:
            x = (x - x_mu) / x.std()

            x = xw^T + b
        Args:
            x: A residual stream tensor of shape (batch, position, d_model)
        Returns:
            A tensor of shape (batch, position, d_model).
        """
        # in practice we can divide by the variance + epsilon (1e-7)
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        return self.weight * x + self.bias


class AttentionBlock(nn.Module):
    """
    Attention block.
    Args:
        n_heads: number of attention heads.
        d_head: model attention head dimension.
        d_model: model residual stream dimension.
    """

    def __init__(self, n_heads: int, d_head: int, d_model: int):
        super().__init__()
        self.d_head = d_head
        self.Q_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.init.xavier_normal_(self.Q_w)
        self.Q_b = nn.Parameter(torch.empty(n_heads, d_head))

        self.K_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.init.xavier_normal_(self.K_w)
        self.K_b = nn.Parameter(torch.empty(n_heads, d_head))

        self.V_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.init.xavier_normal_(self.V_w)
        self.V_b = nn.Parameter(torch.empty(n_heads, d_head))

        self.O_w = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.init.xavier_normal_(self.O_w)
        self.O_b = nn.Parameter(torch.empty(n_heads, d_head))

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies a causal attention mechanism to a residual stream vector `x`.

        Args:
            x: A residual stream tensor of shape (batch, position, d_model)
        Returns:
            A residual stream tensor of shape (batch, position, d_model)
        """
        # the model selects relevant positions by predicting indicies of queries
        queries = einsum(x, self.Q_w, "b pos d_m, head d_m d_h -> b pos head d_h") + self.Q_b
        # ... and the corresponding lookup table to the values
        keys = einsum(x, self.K_w, "b pos d_m, head d_m d_h -> b pos head d_h") + self.K_b

        # applying the attention operation
        attention_logits = einsum(queries, keys, "b q_pos head d_h, b k_pos head d_h -> b head q_pos k_pos")
        attention_logits /= math.sqrt(self.d_head)

        # masking the upper diagonal - the model shouldn't be able to look ahead
        attention_mask = torch.triu(
            torch.ones((attention_logits.shape[-2], attention_logits.shape[-1])),
            diagonal=1,
        )
        attention_masked = attention_logits.masked_fill_(attention_mask.bool().to(attention_logits.device), -1e5)

        # applying softmax so we can find the key with the highest probability
        attention_scores = attention_masked.softmax(-1)

        # the model also has internal feature embeddings for the values at the positions its interested in
        values = einsum(x, self.V_w, "b pos d_m, head d_m d_h -> b pos head d_h") + self.V_b
        # and now selects the values based on the keys which are most relevant to the query
        attention = einsum(
            attention_scores,
            values,
            "b head q_pos k_pos, b k_pos head d_h -> b q_pos head d_h",
        )

        # a final linear projection helps the model decide which subspace in the residual stream to write to.
        return (
            einsum(
                attention,
                self.O_w,
                "b q_pos head d_h, head d_h d_model -> b q_pos d_model",
            )
            + self.O_b
        )


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, activation=F.gelu):
        super().__init__()
        self.In_w = nn.Parameter(torch.empty((d_model, d_mlp)))
        nn.init.xavier_normal_(self.In_w)
        self.In_b = nn.Parameter(torch.empty((d_mlp)))

        self.Out_w = nn.Parameter(torch.empty((d_mlp, d_model)))
        nn.init.xavier_normal_(self.Out_w)
        self.Out_b = nn.Parameter(torch.empty((d_model)))

        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # This block just seems to perform some kind of learned nonlinear operation after attention has been applied to the residual stream.
        x = einsum(x, self.In_w, "b pos d_m, d_m d_mlp -> b pos d_mlp") + self.In_b
        x = self.activation(x)
        x = einsum(x, self.Out_w, "b pos d_mlp, d_mlp d_m -> b pos d_m") + self.Out_b
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block which uses layer norm and attention.
    Args:
        n_heads: number of attention heads.
        d_head: model attention head dimension.
        d_model: model residual stream dimension.
        d_mlp: model MLP dimension.
    """

    def __init__(self, n_heads: int, d_head: int, d_model: int, d_mlp: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attention = AttentionBlock(n_heads, d_head, d_model)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies a decoder-style transformer block to a vector `x`.

        Args:
            x: A residual stream tensor of shape (b, position, d_model)
        """
        residual_norm = self.ln1(x)

        residual_attn = x + self.attention(residual_norm)

        residual_attn_norm = self.ln2(residual_attn)
        return (x + residual_attn) + self.mlp(residual_attn_norm)


class EmbeddingBlock(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.E_w = nn.Parameter(torch.empty((d_vocab, d_model)))
        nn.init.xavier_normal_(self.E_w)

    def forward(self, x: Tensor):
        """
        A lookup table for embedding the vocabulary into the model's space.

        Args:
            x:  Tensor of shape [batch, vocab] where each element is the index in the vocabulary.
        """
        return self.E_w[x, :]


class PositionalEmbeddingBlock(nn.Module):
    def __init__(self, d_model: int, context_length: int):
        super().__init__()
        self.P_w = nn.Parameter(torch.empty((context_length, d_model)))
        nn.init.xavier_normal_(self.P_w)

    def forward(self, x):
        """
        A lookup table for positional embeddings for tokens.

        Args:
            x:  Tensor of shape [batch, vocab] where each element is the index in the vocabulary.
        """
        pos_embed = self.P_w[: x.size(1), :]
        return repeat(pos_embed, "position d_model -> batch position d_model", batch=x.size(0))


class UnembeddingBlock(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.U_w = nn.Parameter(torch.empty((d_model, d_vocab)))
        nn.init.xavier_normal_(self.U_w)
        self.U_b = nn.Parameter(torch.zeros((d_vocab), requires_grad=False))

    def forward(self, x: Tensor):
        """
        Uses a lookup table to convert tokens from model-space to vocabulary-space.

        Args:
            x: a residual stream Tensor of shape [batch position, d_model]
        """
        logits = einsum(x, self.U_w, "batch position d_model, d_model d_vocab -> batch position d_vocab")
        logits += self.U_b
        return logits


class Transformer(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_heads: int,
        d_head: int,
        d_model: int,
        d_mlp: int,
        d_vocab: int,
        context_length: int,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.embedding_block = EmbeddingBlock(d_vocab, d_model)
        self.positional_embedding = PositionalEmbeddingBlock(d_model, context_length)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(n_heads, d_head, d_model, d_mlp) for _ in range(n_blocks)]
        )
        self.unembedding_block = UnembeddingBlock(d_vocab, d_model)
        self.ln = LayerNorm(d_model)
        self.device = device

    def to(self, device):
        self.device = device
        super().to(device)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.embedding_block(x) + self.positional_embedding(x)
        for layer in self.transformer_blocks:
            residual = layer(residual)
        residual = self.ln(residual)
        return self.unembedding_block(residual)

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            rearrange(input, "b pos vocab -> (b pos) vocab"), rearrange(target, "b pos -> (b pos)"), ignore_index=0
        )

    def train_forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self(x)
        loss = self.loss(logits, y)
        return loss, logits
