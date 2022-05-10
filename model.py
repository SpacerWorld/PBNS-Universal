import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, shape, activation=None, name='fc', **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.w = tf.Variable(tf.initializers.glorot_normal()(shape), name=name + '_w')
    self.b = tf.Variable(tf.zeros(shape[-1], dtype=tf.float32), name=name + '_b')
    self.activation = activation or (lambda x: x)

  def call(self, x):
    x = tf.einsum('ab,bc->ac', x, self.w) + self.b
    x = self.activation(x)
    return x

class PBNS(tf.keras.Model):
  def __init__(self, num_vertices):
    super(PBNS, self).__init__()
    self.encoder = [
        Encoder(shape=(79,  256), activation=tf.nn.relu),
        Encoder(shape=(256, 256), activation=tf.nn.relu),
        Encoder(shape=(256, 256), activation=tf.nn.relu),
        Encoder(shape=(256, 256), activation=tf.nn.relu)
    ]
    shape = (self.encoder[-1].w.shape[-1], num_vertices, 3)
    self.psd = tf.Variable(tf.initializers.glorot_normal()(shape), name='psd')

  def gather(self):
    weights = [self.psd]
    for encoder in self.encoder:
      weights += [encoder.w, encoder.b]
    return weights

  def call(self, x):
    for encoder in self.encoder:
      x = encoder(x)
    return tf.einsum('ab,bcd->acd', x, self.psd)
