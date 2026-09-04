# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Encoder-Transformer-Decoder graph net model."""

import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf
from .TransformerBlock import TransformerBlock

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple( 'Graph', ['node_features', 'edge_sets', 'positions'])



# ---------------------------------------------------------------------------
# Graph-aware Transformer processor block
# ---------------------------------------------------------------------------

class GraphTransformerBlock(snt.AbstractModule):
  """Transformer block that is aware of graph edges.

  For each edge set, it:
    1. Updates edge features using sender/receiver node features (as before).
    2. Aggregates updated edge features into nodes.
  Then applies a Transformer layer over the combined node representation.
  """

  def __init__(self, model_fn, model_dim, num_heads, ffn_dim,
               name='GraphTransformerBlock'):
    super(GraphTransformerBlock, self).__init__(name=name)
    self._model_fn = model_fn
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._ffn_dim = ffn_dim

  def _update_edge_features(self, node_features, edge_set):
    """Gather node features and compute new edge features via MLP."""
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    with tf.variable_scope(edge_set.name + '_edge_fn'):
      return self._model_fn()(tf.concat(features, axis=-1))

  def _aggregate_edge_to_node(self, node_features, edge_sets):
    """Aggregate edge messages into nodes and concatenate with node features."""
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(
          tf.math.unsorted_segment_sum(
              edge_set.features, edge_set.receivers, num_nodes))
    with tf.variable_scope('node_proj'):
      return self._model_fn()(tf.concat(features, axis=-1))

  def _build_attention_bias(self, num_nodes, edge_sets):
    """Build a [N, N] additive attention bias from edge connectivity.

    Connected pairs get bias=0, unconnected pairs get a large negative value
    so that attention is guided by the graph topology.
    """
    # Start with a large negative value (masking unconnected pairs)
    neg_inf = tf.fill([num_nodes, num_nodes], -1e9)
    # Allow self-attention
    diag = tf.eye(num_nodes) * 1e9          # undo mask on diagonal
    bias = neg_inf + diag
    for edge_set in edge_sets:
      # Mark connected pairs with bias=0 (no penalty)
      indices = tf.stack([edge_set.receivers, edge_set.senders], axis=1)
      updates = tf.zeros([tf.shape(edge_set.senders)[0]], dtype=tf.float32)
      bias = tf.tensor_scatter_nd_update(bias, indices, updates)
    return bias  # [N, N]

  def _build(self, graph, is_training):
    """Apply GraphTransformerBlock and return updated MultiGraph."""
    # 1. Update edge features (MLP on sender/receiver/edge)
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated = self._update_edge_features(graph.node_features, edge_set)
      new_edge_sets.append(edge_set._replace(features=updated))

    # 2. Aggregate edge messages into nodes
    aggregated_node_features = self._aggregate_edge_to_node(
        graph.node_features, new_edge_sets)

    # 3. Build adjacency-aware attention bias
    num_nodes = tf.shape(graph.node_features)[0]
    attn_bias = self._build_attention_bias(num_nodes, new_edge_sets)

    # 4. Apply Transformer layer over node features
    with tf.variable_scope('transformer'):
      new_node_features = TransformerBlock(
          self._model_dim, self._num_heads, ffn_dim=self._model_dim * 4)(
          aggregated_node_features,
          positions=graph.positions, 
          attention_bias=attn_bias,
          is_training = is_training)

    # 5. Residual connections (same as original GraphNetBlock)
    new_node_features += graph.node_features
    new_edge_sets = [
        es._replace(features=es.features + old_es.features)
        for es, old_es in zip(new_edge_sets, graph.edge_sets)
    ]

    return MultiGraph(new_node_features, new_edge_sets, graph.positions)


# ---------------------------------------------------------------------------
# Top-level Encode-Process(Transformer)-Decode model
# ---------------------------------------------------------------------------

class EncodeTransformerDecode(snt.AbstractModule):
  """Encode-Transformer-Decode GraphNet model.

  Replaces the message-passing GraphNetBlock processor with a stack of
  graph-aware Transformer blocks while keeping the same encoder/decoder
  structure as the original EncodeProcessDecode.

  Args:
    output_size          : int, dimensionality of decoded output per node.
    latent_size          : int, width of hidden layers and attention dim D.
    num_layers           : int, number of MLP layers inside each MLP.
    message_passing_steps: int, number of Transformer processor steps.
    num_heads            : int, number of attention heads (must divide latent_size).
  """

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               num_heads=4,
               name='EncodeTransformerDecode'):
    super(EncodeTransformerDecode, self).__init__(name=name)
    assert latent_size % num_heads == 0, (
        f'latent_size ({latent_size}) must be divisible by num_heads ({num_heads})')
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self._num_heads = num_heads

  # ---- helpers ----

  def _make_mlp(self, output_size, layer_norm=True):
    """Builds an MLP with optional LayerNorm."""
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  # ---- encoder ----

  def _encoder(self, graph):
    """Projects raw node/edge features into the latent space."""
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      new_edge_sets = []
      for edge_set in graph.edge_sets:
        latent = self._make_mlp(self._latent_size)(edge_set.features)
        new_edge_sets.append(edge_set._replace(features=latent))
    return MultiGraph(node_latents, new_edge_sets, positions=graph.positions)

  # ---- transformer processor ----

  def _processor(self, latent_graph, is_training):
    """Stacks GraphTransformerBlocks as the processor."""
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    for step in range(self._message_passing_steps):
      with tf.variable_scope(f'transformer_step_{step}'):
        latent_graph = GraphTransformerBlock(
            model_fn=model_fn,
            model_dim=self._latent_size,
            num_heads=self._num_heads,
            ffn_dim=self._latent_size * 4
        )(latent_graph, is_training=is_training)
    return latent_graph

  # ---- decoder ----

  def _decoder(self, graph):
    """Projects node latents to the output space."""
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp(self._output_size, layer_norm=False)
      return decoder(graph.node_features)

  # ---- forward pass ----

  def _build(self, graph, is_training):
    """Encodes, processes with Transformer, decodes, returns node features."""
    latent_graph = self._encoder(graph)
    latent_graph = self._processor(latent_graph, is_training)
    return self._decoder(latent_graph)