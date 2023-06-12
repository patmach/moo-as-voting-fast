import numpy as np
x = np.array([[[0]], [[1]], [[2]]])
print(x.shape)

print(np.squeeze(x).shape)

#np.squeeze(x, axis=0).shape

print(np.squeeze(x, axis=1).shape)

print(np.squeeze(x, axis=2).shape)