
import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf
from .MultiHeadSelfAttention import MultiHeadSelfAttention

# ---------------------------------------------------------------------------
# Transformer block (self-attention + FFN + residual + layer-norm)
# ---------------------------------------------------------------------------
class GeometricPositionalEncoding(snt.AbstractModule):
    """
    Graph positional encoding using 3D geometry projection.
    Suitable for mesh or spatial graph nodes.
    """

    def __init__(self, model_dim, name='GeometricPositionalEncoding'):
        super().__init__(name=name)
        self._model_dim = model_dim

    def _build(self, positions):
        """
        positions: [N, 3]
        Returns:
            positional embedding: [N, model_dim]
        """

        with tf.variable_scope('gpe'):

            linear_proj = snt.Linear(self._model_dim)
            pos_embedding = linear_proj(positions)

            return pos_embedding


class TransformerBlock(snt.AbstractModule):
  """One Transformer encoder layer applied over node tokens.

  Uses pre-norm (LayerNorm before sub-layer) for training stability.
  """

  def __init__(self, model_dim, num_heads, ffn_dim, dropout_rate=0.1, use_gpe=True, name='TransformerBlock'):
    super(TransformerBlock, self).__init__(name=name)
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._ffn_dim = ffn_dim
    self._dropout_rate = dropout_rate
    self._use_gpe = use_gpe

    if use_gpe:
          self._gpe = GeometricPositionalEncoding(model_dim)

  def _build(self, x, positions=None, attention_bias=None,  is_training=None):
    """
    Args:
      x              : [N, D] node (or edge) feature tensor
      attention_bias : optional [N, N] additive bias for attention logits
    Returns:
      [N, D] updated features
    """
    # ===============================
    # Graph Positional Encoding
    # ===============================
    if self._use_gpe and positions is not None:
            with tf.variable_scope('gpe_block'):
                pe = self._gpe(positions)
                x = x + pe
                x = tf.layers.dropout( x, rate=self._dropout_rate, training=is_training)

    # --- Self-attention sub-layer (pre-norm) ---
    with tf.variable_scope('attn'):
      x_norm = snt.LayerNorm()(x)
      attn_out = MultiHeadSelfAttention(self._num_heads, self._model_dim)(
          x_norm, bias=attention_bias)
       # Dropout BEFORE residual
      attn_out = tf.layers.dropout(
          attn_out, rate=self._dropout_rate,training=is_training)
      
      x = x + attn_out  # residual

     # ===============================
     # Feed Forward Network
    # ===============================
    with tf.variable_scope('ffn'):
      x_norm = snt.LayerNorm()(x)

      ffn = snt.nets.MLP(
          [self._ffn_dim, self._model_dim],
          activate_final=False,
          activation=tf.nn.relu,
      )
      ffn_out = ffn(x_norm)
      
   # Dropout BEFORE residual
      ffn_out = tf.layers.dropout(
          ffn_out, rate=self._dropout_rate,training=is_training)

      x = x + ffn_out # residual

    return x

