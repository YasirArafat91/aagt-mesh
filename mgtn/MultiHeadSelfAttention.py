



import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf

# ---------------------------------------------------------------------------
# Utility: Multi-Head Self-Attention (Sonnet / TF1 compatible)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(snt.AbstractModule):
  """Scaled dot-product multi-head self-attention over a set of tokens.

  Input shape : [N, D]  (N nodes / edges treated as sequence tokens)
  Output shape: [N, D]
  """

  def __init__(self, num_heads, model_dim, name='MultiHeadSelfAttention'):
    super(MultiHeadSelfAttention, self).__init__(name=name)
    assert model_dim % num_heads == 0, 'model_dim must be divisible by num_heads'
    self._num_heads = num_heads
    self._model_dim = model_dim
    self._head_dim = model_dim // num_heads

  def _build(self, x, bias=None):
    """
    Args:
      x   : [N, D] float tensor
      bias: optional [N, N] additive attention bias (e.g. adjacency mask)
    Returns:
      [N, D] float tensor
    """
    d = self._model_dim
    h = self._num_heads
    hd = self._head_dim
    n = tf.shape(x)[0]

    # Linear projections – shared across the whole block
    Wq = tf.get_variable('Wq', shape=[d, d], initializer=tf.glorot_uniform_initializer())
    Wk = tf.get_variable('Wk', shape=[d, d], initializer=tf.glorot_uniform_initializer())
    Wv = tf.get_variable('Wv', shape=[d, d], initializer=tf.glorot_uniform_initializer())
    Wo = tf.get_variable('Wo', shape=[d, d], initializer=tf.glorot_uniform_initializer())

    Q = tf.matmul(x, Wq)  # [N, D]
    K = tf.matmul(x, Wk)
    V = tf.matmul(x, Wv)

    # Split heads: [N, D] -> [h, N, hd]
    def split_heads(t):
      t = tf.reshape(t, [n, h, hd])   # [N, h, hd]
      return tf.transpose(t, [1, 0, 2])  # [h, N, hd]

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

    # Scaled dot-product attention
    scale = tf.cast(hd, tf.float32) ** -0.5
    logits = tf.matmul(Q, K, transpose_b=True) * scale  # [h, N, N]
    if bias is not None:
      logits += tf.expand_dims(bias, 0)  # broadcast over heads
    weights = tf.nn.softmax(logits, axis=-1)              # [h, N, N]
    attended = tf.matmul(weights, V)                      # [h, N, hd]

    # Merge heads: [h, N, hd] -> [N, D]
    attended = tf.transpose(attended, [1, 0, 2])          # [N, h, hd]
    attended = tf.reshape(attended, [n, d])               # [N, D]

    return tf.matmul(attended, Wo)

