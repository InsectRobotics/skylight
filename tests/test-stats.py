import numpy as np
import matplotlib.pyplot as plt
from learn import rad2compass


src = np.load('seville-bb-4-20170621.npz')
x = src['x'].reshape((-1, 473))
mu = x.mean(axis=0)
x0 = x - mu
C = x0.T.dot(x0)
U, S, V = np.linalg.svd(C)

plt.subplot(2, 2, 1)
plt.imshow(U, vmin=-1, vmax=1, cmap="coolwarm")
plt.title("U")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(V, vmin=-1, vmax=1, cmap="coolwarm")
plt.title("V")
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2, 1, 2)
plt.plot(np.arange(358, 831), S)
plt.title("S")
plt.ylim([0, 0.002])
plt.grid()

plt.show()
