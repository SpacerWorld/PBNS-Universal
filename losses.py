import sys
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.representation as R
from scipy.spatial import cKDTree

def loss_fn(x, body, faces, options, config,
    edge_young=5, bend_weight=5*1e-5, collision_weight=25, mass=0.3):
  edge, error_edge = edge_loss(x, options['computed_edges'], options['edges'], weights=config['edge'])
  bend, error_bend = bend_loss(x, options['faces'], options['neigh_faces'], weights=config['bend'])
  collision, error_collision = collision_loss(x, options['faces'], body, faces, config['layers'], thr=0.004)
  gravity, error_gravity = gravity_loss(x, options['computed_area'], m=0.3)
  loss = edge_young * edge + bend_weight * bend + collision_weight * collision + gravity
  return loss, {
    'edge': error_edge.numpy(),
    'bend': error_bend.numpy(),
    'collision': error_collision,
    'gravity': error_gravity.numpy()
  }

def edge_loss(x, y_e, E, weights):
  x_e = tf.gather(x, E[:,0], axis=1) - tf.gather(x, E[:,1], axis=1)
  x_e = tf.sqrt(tf.reduce_sum(x_e ** 2, -1))
  d_e = (x_e - y_e[None])
  err = tf.reduce_mean(tf.abs(d_e))
  d_e = d_e ** 2
  d_e = tf.multiply(weights[None], d_e)
  loss = tf.reduce_sum(d_e)
  return loss, err

def bend_loss(x, F, neighF, weights):
  # compute face normals
  V02 = tf.gather(x, F[:,2], axis=1) - tf.gather(x, F[:,0], axis=1) # B x N x 3
  V01 = tf.gather(x, F[:,1], axis=1) - tf.gather(x, F[:,0], axis=1) # B x N x 3
  N = tf.linalg.cross(V02, V01) # B x N x 3
  N = N / tf.linalg.norm(N, axis=-1, keepdims=True) # B x N x 3
  # compare neighbouring face normals
  N0 = tf.gather(N, neighF[:,0], axis=1) # B x N x 3
  N1 = tf.gather(N, neighF[:,1], axis=1) # B x N x 3
  # cosinus distances
  cos = tf.reduce_sum(tf.multiply(N0, N1), -1)
  err = tf.reduce_mean(1 - cos)
  dN = (N0 - N1) ** 2
  dN = tf.multiply(weights[None,:,None], dN)
  loss = tf.reduce_sum(dN)
  return loss, err

def collision_loss(V, F, B, B_F, layers, thr=.004, stop_gradient=False):
  V_vn = R.mesh.normals.vertex_normals(V, tf.tile(F[None], [V.shape[0], 1, 1]))
  B_vn = R.mesh.normals.vertex_normals(B, tf.tile(B_F[None], [B.shape[0], 1, 1]))
  loss = 0
  vcount = np.array([0] * len(layers), np.float32)
  for i in range(len(layers)):
    l = layers[i]
    # get layer verts
    v = tf.gather(V, l, axis=1)
    # compute correspondences
    idx = _tf_nn_parallel(v, B)
    N, M = v.shape[:2]
    _idx = tf.tile(np.array(list(range(N)), np.int32)[:,None], [1, M])
    idx = tf.stack((_idx, idx), axis=-1)
    # compute loss
    D = v - tf.gather_nd(B, idx)
    b_vn = tf.gather_nd(B_vn, idx)
    dot = tf.einsum('abc,abc->ab', D, b_vn)
    loss += tf.reduce_sum(tf.minimum(dot - thr, 0) ** 2)
    # vcount
    vmask = tf.cast(tf.math.less(dot, 0), tf.float32)
    vcount[i] = tf.reduce_sum(vmask).numpy() / (N * M)
    # add layer to "body"
    if i + 1 < len(layers):
      if stop_gradient:
        v = tf.stop_gradient(v)
        v_vn = tf.stop_gradient(tf.gather(V_vn, l, axis=1))
      else:
        v_vn = tf.gather(V_vn, l, axis=1)
      B = tf.concat((B, v), axis=1)
      B_vn = tf.concat((B_vn, v_vn), axis=1)
  return loss, vcount

def _nearest_neighbour(V, B):
  tree = cKDTree(B.numpy())
  return tree.query(V.numpy(), n_jobs=-1)[1]

@tf.function
def _tf_nn_parallel(V, B):
  return tf.map_fn(fn=lambda elem: tf.py_function(func=_nearest_neighbour, inp=elem, Tout=tf.int32), elems=(V,B), fn_output_signature=tf.int32, parallel_iterations=V.shape[0])

def gravity_loss(x, surface, m=.15, g=9.81):
  vertex_mass = surface * m / x.shape[1]
  U = vertex_mass * g * x[:,:,2]
  return tf.reduce_sum(U), tf.reduce_mean(tf.reduce_sum(U, -1))
