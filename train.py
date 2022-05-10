import rw, os
import numpy as np
from tqdm import tqdm
from model import PBNS
from smpl import SMPL, LBS
import tensorflow as tf
from losses import loss_fn
from utils import (
  make_options,
  make_configs,
  find_nearest_neighbour
)

epochs = 100
batch_size = 16

if __name__ == "__main__":
  if not os.path.isdir("result"):
    os.mkdir("result")

  garment, faces = rw.read("assets/garment.obj")
  options        = make_options(garment, faces)
  config         = make_configs(garment, options)
  garment        = tf.convert_to_tensor(garment, dtype=tf.float32)
  SMPL           = SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
  data           = tf.data.Dataset.from_tensor_slices(np.load("assets/train.npy").astype(np.float32))
  data           = data.shuffle(buffer_size=3)
  data           = data.batch(batch_size=16)
  model          = PBNS(num_vertices=garment.shape[0])

  body, params   = SMPL(shape=tf.zeros(shape=[1, 10]), pose=tf.zeros(shape=[1, 72]))
  indices        = find_nearest_neighbour(garment, body[0])
  weights        = tf.gather(SMPL.skinning_weights, indices)

  trainables     = model.gather()
  optimizer      = tf.optimizers.SGD(learning_rate=1e-5, momentum=0.9)

  for epoch in range(epochs):
    metrices = {}
    # train
    for poses in tqdm(data):
      with tf.GradientTape() as tape:
        shapes         = tf.random.uniform(shape=[poses.shape[0], 10], minval=-5, maxval=5, dtype=tf.float32)
        v_body, params = SMPL(shape=shapes, pose=poses)
        features       = tf.concat([poses[:, 3:], shapes], axis=1)
        v_deformed     = garment[None] + model(features)
        v_garment      = LBS()(v_deformed, params["J_transforms"], weights)
        loss, errors   = loss_fn(v_garment, v_body, SMPL.faces, options, config)

      grads = tape.gradient(loss, trainables)
      optimizer.apply_gradients(zip(grads, trainables))

      # update metrices
      for key in errors:
        if not key in metrices:
          metrices[key] = errors[key]
        else:
          metrices[key] += errors[key]
    data.shuffle(buffer_size=epoch + 1)
    rw.write("result/body.obj", body[0].numpy(), SMPL.faces.numpy())
    rw.write("result/garment.obj", garment.numpy(), faces)

    if os.path.isfile(os.path.join("result", "body.pc2")):
      os.remove(os.path.join("result", "body.pc2"))

    if os.path.isfile(os.path.join("result", "garment.pc2")):
      os.remove(os.path.join("result", "garment.pc2"))

    # eval
    test = np.load("assets/test.npy").astype(np.float32)
    n_points = int(np.ceil(test.shape[0] / 10))
    shapes = []
    for i in range(n_points + 1):
      shapes += [3 * np.random.uniform(-1, 1, size=(10,))]
    shapes = np.array(shapes)
    shapes = tf.convert_to_tensor(
      np.concatenate([np.linspace(shapes[i], shapes[i + 1], 10) for i in range(shapes.shape[0] - 1)], axis=0),
      dtype=tf.float32)

    test = tf.data.Dataset.from_tensor_slices(test)
    test = test.batch(batch_size=16)
    i = 0
    for poses in tqdm(test):
      v_body, params = SMPL(shape=shapes[i:i + poses.shape[0]], pose=poses)
      features       = tf.concat([poses[:, 3:], shapes[i:i + poses.shape[0]]], axis=1)
      v_deformed     = garment[None] + model(features)
      v_garment      = LBS()(v_deformed, params["J_transforms"], weights)
      rw.write("result/body.pc2", v_body.numpy())
      rw.write("result/garment.pc2", v_garment.numpy())
      i += poses.shape[0]

    # display metrices
    print('epoch', epoch + 1)
    for key in metrices:
      print(key, metrices[key])
