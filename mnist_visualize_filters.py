"""
visualize filters
"""
import os
import keras
import matplotlib.pyplot as plt

from mnist_layers import LinearEncodeModel

weights_enc = "weights_enc.hdf5"

# encoder
model_enc = LinearEncodeModel()
if os.path.exists(weights_enc):
    model_enc.load_weights(weights_enc)

# display
weights,bias = model_enc.layers[-1].get_weights()

for i in range(weights.shape[-1]):
    img = weights[:,i].reshape((28,28))
    img = 3*img+0.5

    plt.subplot(4,8,i+1)
    plt.tick_params(labelbottom="off",bottom="off",labelleft="off",left="off")
    plt.imshow(img, 'bwr', vmin=0, vmax=1)

plt.savefig("components.png")