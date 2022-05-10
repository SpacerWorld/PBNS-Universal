import os
import json
import numpy as np
from struct import pack, unpack

def read_json(path):
  with open(path) as __file:
    return json.load(__file)

def read_obj(path):
  V = []
  F = []
  with open(path, 'r') as __file:
    T = __file.readlines()
  for line in T:
    if line.startswith('v '):
      V.append([float(n) for n in line.replace('v ','').split(' ')])
    elif line.startswith('f '):
      try:
        F.append([int(n) - 1 for n in line.replace('f ','').split(' ')])
      except:
        try:
          F.append([int(n.split('//')[0]) - 1 for n in line.replace('f ','').split(' ')])
        except:
          F.append([int(n.split('/')[0]) - 1 for n in line.replace('f ','').split(' ')])
    elif line.startswith('l '):
      F.append([int(n) - 1 for n in line.replace('l ','').split(' ')])
  return np.array(V, np.float32), F

def write_obj(path, vertices, faces):
  with open(path, 'w') as __file:
    __file.write('s 1\n')
    for vertex in vertices:
      line = 'v {}\n'.format(' '.join([str(_) for _ in vertex]))
      __file.write(line)
    for face in faces:
      line = 'f {}\n'.format(' '.join([str(_ + 1) for _ in face]))
      if len(face) == 2:
        line = line.replace('f ', 'l ')
      __file.write(line)

def write_pc2(path, V, float16=False):
  if float16: V = V.astype(np.float16)
  else: V = V.astype(np.float32)
  with open(path, 'wb') as __file:
    header_format='<12siiffi'
    header_str = pack(header_format, b'POINTCACHE2\0', 1, V.shape[1], 0, 1, V.shape[0])
    __file.write(header_str)
    __file.write(V.tobytes())

def write_pc2_frames(path, V, float16=False):
  if os.path.isfile(path):
    if float16: V = V.astype(np.float16)
    else: V = V.astype(np.float32)
    with open(path, 'rb+') as __file:
      __file.seek(16)
      nPoints = unpack('<i', __file.read(4))[0]
      assert len(V.shape) == 3 and V.shape[1] == nPoints, 'Inconsistent dimensions: ' + str(V.shape) + ' and should be (-1,' + str(nPoints) + ',3)'
      __file.seek(28)
      nSamples = unpack('<i', __file.read(4))[0]
      nSamples += V.shape[0]
      __file.seek(28)
      __file.write(pack('i', nSamples))
      __file.seek(0, 2)
      __file.write(V.tobytes())
  else: write_pc2(path, V, float16)

def read_pc2(path):
  data = {}
  bytes = 4
  dtype = np.float32
  with open(path, 'rb') as __file:
    data['sign'] = __file.read(12)
    data['version'] = unpack('<i', __file.read(4))[0]
    data['nPoints'] = unpack('<i', __file.read(4))[0]
    data['startFrame'] = unpack('f', __file.read(4))
    data['sampleRate'] = unpack('f', __file.read(4))
    data['nSamples'] = unpack('<i', __file.read(4))[0]
    size = data['nPoints'] * data['nSamples'] * 3 * bytes
    data['V'] = np.frombuffer(__file.read(size), dtype=dtype).astype(np.float32)
    data['V'] = data['V'].reshape(data['nSamples'], data['nPoints'], 3)
  return data['V']

def read(path):
  if path.endswith(".obj"):
    return read_obj(path)
  if path.endswith(".json"):
    return read_json(path)
  if path.endswith(".pc2"):
    return read_pc2(path)

def write(path, *params):
  if path.endswith(".obj"):
    return write_obj(path, *params)
  
  if path.endswith(".pc2"):
    return write_pc2_frames(path, *params)
