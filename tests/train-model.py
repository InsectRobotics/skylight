import numpy as np
from learn import from_file
import matplotlib.pyplot as plt


model_name = "seville-cr-jun"
# names = ["seville-bb-4-20170321", "seville-bb-4-20170621", "seville-bb-4-20170921", "seville-bb-4-20171221"]
names = ["seville-cr-32-20170621", "seville-cr-32-20170601"]

model = from_file("chrom-mon-rnn.yaml")
model.compile(optimizer="rmsprop", loss="mae", metrics=["accuracy"])
model.summary()
# model.load_weights("%s.h5" % model_name)

loss, acc = model.train([names[0]], valid_data=[names[1]])

# plot progress
plt.figure(1, figsize=(15, 20))

plt.subplot(121)
plt.plot(loss)
plt.ylim([0, 1])

plt.subplot(122)
plt.plot(acc)
plt.ylim([0, 1])

plt.show()