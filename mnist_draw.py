import os
import sys
import keras
from keras.datasets import mnist
from keras.models import Model, Input 
from keras.optimizers import RMSprop

from mnist_layers import DrawImageLayer,LinearEncodeModel,LSTMDecodeModel

batch_size = 256
epochs = int(sys.argv[1]) # set 0 to display sequence
steps = 50

weights_enc = "weights_enc.hdf5"
weights_dec = "weights_dec.hdf5"
train = True if epochs>0 else False
sequence = not train


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model_enc = LinearEncodeModel()
model_dec = LSTMDecodeModel(steps)

## main model
x = Input(shape=(28,28,1))
x_enc = model_enc(x)
x_dec = model_dec(x_enc)
x_image = DrawImageLayer(return_sequences=sequence)(x_dec)
model = Model(inputs=[x],outputs=[x_image])
model.summary()

if os.path.exists(weights_enc):
    model_enc.load_weights(weights_enc)

if os.path.exists(weights_dec):
    model_dec.load_weights(weights_dec)

if train:
    model.compile(loss='mse',optimizer=RMSprop(0.0015))
    history = model.fit(x_train, x_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, x_test))

    model_enc.save_weights(weights_enc)
    model_dec.save_weights(weights_dec)

# # export to png file with matplotlib
# import matplotlib.pyplot as plt

# img0 = x_test[:100]
# img1 = model.predict(img0.reshape((-1,28,28,1)))

# for i in range(100):
#     plt.subplot(10,10,i+1)
#     plt.tick_params(labelbottom="off",bottom="off",labelleft="off",left="off")
#     plt.imshow(img0[i,:,:,0], 'gray', vmin=0, vmax=1)
# plt.savefig("true.png")

# for j in range(steps):
#     for i in range(100):
#         plt.subplot(10,10,i+1)
#         plt.tick_params(labelbottom="off",bottom="off",labelleft="off",left="off")
#         plt.imshow(img1[i,j,:,:,0], 'gray', vmin=0, vmax=1)
#     plt.savefig("draw{:0>2}.png".format(j))


# display with opencv
import cv2
for i in range(100):
    img0 = x_test[i]
    img1 = model.predict(img0.reshape((1,28,28,1)))[0]
    
    if len(img1.shape)==4:
        for j in range(img1.shape[0]):

            cv2.imshow("gt",img0)
            cv2.imshow("predict",img1[j])
            cv2.waitKey(15)
        cv2.imshow("gt",img0)
        cv2.imshow("predict",img1[-1])
        k = cv2.waitKey()
        if k==27:
            break

    else:
        cv2.imshow("gt",img0)
        cv2.imshow("predict",img1)
        k = cv2.waitKey()
        if k==27:
            break