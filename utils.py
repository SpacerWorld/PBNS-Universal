import sys
import numpy as np
import tensorflow as tf

def pairwise_distance(A, B):
  rA = np.sum(np.square(A), axis=1)
  rB = np.sum(np.square(B), axis=1)
  distances = -2 * np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
  return distances

def find_nearest_neighbour(A, B, dtype=np.int32):
  nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
  return nearest_neighbour.astype(dtype)

def quads2tris(F):
  F_out = []
  for f in F:
    if len(f) <= 3: F_out += [f]
    elif len(f) == 4:
      F_out += [
        [f[0], f[1], f[2]],
        [f[0], f[2], f[3]]
      ]
    else:
      print('This should not happen, but might')
      print('To solve: extend this to deal with 5-gons or ensure mesh is quads/tris only')
      sys.exit()
  return np.array(F_out, np.int32)

def faces2edges(F):
  E = set()
  for f in F:
    N = len(f)
    for i in range(N):
      j = (i + 1) % N
      E.add(tuple(sorted([f[i], f[j]])))
  return np.array(list(E), np.int32)

def make_neigh_faces(F, E=None):
  if E is None: E = faces2edges(F)
  G = {tuple(e): [] for e in E}
  for i,f in enumerate(F):
    n = len(f)
    for j in range(n):
      k = (j + 1) % n
      e = tuple(sorted([f[j], f[k]]))
      G[e] += [i]
  neighF = []
  for key in G:
    if len(G[key]) == 2:
      neighF += [G[key]]
    elif len(G[key]) > 2:
      print("Neigh F unexpected behaviour")
      continue
  return np.array(neighF, np.int32)

def compute_edges(v_template, edges):
  computed_edges = tf.gather(v_template, edges[:, 0], axis=0) - tf.gather(v_template, edges[:, 1], axis=0)
  computed_edges = tf.sqrt(tf.reduce_sum(computed_edges ** 2, -1))
  return compute_edges

def compute_area(SMGL, options):
  v_template = SMGL.v_template
  faces = options['faces']
  u = tf.gather(v_template, faces[:,2], axis=0) - tf.gather(v_template, faces[:,0], axis=0)
  v = tf.gather(v_template, faces[:,1], axis=0) - tf.gather(v_template, faces[:,0], axis=0)
  computed_area = tf.norm(tf.linalg.cross(u, v), axis=-1)
  computed_area = tf.reduce_sum(computed_area) / 2.0
  return computed_area

def make_options(garment, faces):
  faces = quads2tris(faces)
  edges = faces2edges(faces)
  neigh_faces = make_neigh_faces(faces, edges)

  computed_edges = tf.gather(garment, edges[:, 0], axis=0) - tf.gather(garment, edges[:, 1], axis=0)
  computed_edges = tf.sqrt(tf.reduce_sum(computed_edges ** 2, -1))

  u = tf.gather(garment, faces[:,2], axis=0) - tf.gather(garment, faces[:,0], axis=0)
  v = tf.gather(garment, faces[:,1], axis=0) - tf.gather(garment, faces[:,0], axis=0)
  computed_area = tf.norm(tf.linalg.cross(u, v), axis=-1)
  computed_area = tf.reduce_sum(computed_area) / 2.0

  return {
    'computed_edges': computed_edges,
    'computed_area': computed_area,
    'edges': edges,
    'faces': faces,
    'neigh_faces': neigh_faces
  }

def make_configs(garment, options):
  N = garment.shape[0]
  config = {
    'layers': [list(range(N))],
    'edge': np.ones((N,), np.float32),
    'bend': np.ones((N,), np.float32),
  }

  edge_weights = np.zeros((len(options['edges']),), np.float32)
  for i, e in enumerate(options['edges']):
    edge_weights[i] = config['edge'][e].mean()
  config['edge'] = edge_weights

  bend_weights = np.zeros((len(options['neigh_faces']),), np.float32)
  for i, n_f in enumerate(options['neigh_faces']):
    v = list(set(options['faces'][n_f[0]]).intersection(set(options['faces'][n_f[1]])))
    bend_weights[i] = config['bend'][v].mean()
  config['bend'] = bend_weights
  return config
