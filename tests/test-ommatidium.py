import numpy as np
import matplotlib.pyplot as plt

from sky import ChromaticitySkyModel, Panorama
from datetime import datetime


sky = ChromaticitySkyModel(turbidity=2, nside=1)
sky.obs.date = datetime(2017, 06, 21, 9, 0, 0)
sky.generate(show=False)

p = Panorama(200, sky)
p.update()

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(p._img)
plt.title("Colour")
print p._img.max(), p._img.min()

plt.subplot(132)
plt.imshow(p._dop)
plt.title("DOP")
print p._dop.max(), p._dop.min()

plt.subplot(133)
plt.imshow(p._aop)
plt.title("AOP")
print p._aop.max(), p._aop.min()

plt.show()
