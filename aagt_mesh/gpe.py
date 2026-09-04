
import sonnet as snt
import tensorflow.compat.v1 as tf

class GeometricPositionalEncoding(snt.AbstractModule):

    def __init__(self, d_model, name="geometric_positional_encoding"):
        super(GeometricPositionalEncoding, self).__init__(name=name)
        self._d_model = d_model

    def _build(self, positions):

        mlp = snt.nets.MLP(
            output_sizes=[self._d_model, self._d_model],
            activation=tf.nn.relu,
            name="mlp_projection"
        )

        return mlp(positions)


# class GeometricPositionalEncoding(object):
#     """
#     Geometric positional encoding from mesh node coordinates.
#     """

#     def __init__(self, d_model):
#         self.linear = tf.compat.v1.layers.Dense(d_model)

#     def __call__(self, positions):
#         # positions shape: [N, 3]
#         return self.linear(positions)
