import numpy as np
import pandas as pd


def eucledian_distance(point1, point2):
    dist = 0
    for i in range(len(point1)):
      dist += ((point1[i] - point2[i])**2)
    return np.sqrt(dist)

def train(x, y, z, k):
  distancia = []
  for i in range(x.shape[0]):
    distancia.append(eucledian_distance(z, x[i,]))
  s = np.argsort(distancia)[0:k]
  u = np.bincount(y[s])
  m = []
  max_value = max(u)
  for i, x in enumerate(u):
    if x == max_value:
      m.append(i)  
  while (len(m)>1):
    k = k -1
    s = s[0:k]
    u = np.bincount(y[s])
    m = []
    max_value = max(u)
    for i, x in enumerate(u):
      if x == max_value:
        m.append(i)  
  return m[0]
def vizinhos(x,y,z,k):
  w=[]
  for i in range(z.shape[0]):
    w.append(train(x,y,z[i,],k))
  return w
