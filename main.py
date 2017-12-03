
# coding: utf-8

# In[1]:

from ChannelEstimatorNetwork import SparseEstimatorNetwork
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import tensorflow as tf
import os

#matplotlib.use('Qt5Agg')


flags = tf.app.flags
flags.DEFINE_string("dataset", "", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("logdir", "", "Directory name to save the image samples [output]")
FLAGS = flags.FLAGS


# In[7]:
#def main(_):
on_cloud=1

if (on_cloud == 1):
    log_path = os.path.join("/output/",FLAGS.logdir)
    #data_path = os.path.join("/data/",FLAGS.dataset)
    data_path = os.path.join(FLAGS.dataset)
else:
    log_path = os.path.join("./output/",FLAGS.logdir)
    #data_path = os.path.join(FLAGS.dataset)
    data_path = os.path.join("/Local_data/chimage_data_2/","")


Data_file=data_path+"/Ch_real_VehA_14.mat"


channels = scipy.io.loadmat(Data_file)['channels']
reals = np.real(channels)
imags = np.imag(channels)
all_channel_images = np.vstack([reals, imags])
all_channel_images = (all_channel_images+5)/10.0

#print(np.amax(all_channel_images))

#print(np.amin(all_channel_images))

#exit()

X_train , X_test = train_test_split(all_channel_images, test_size=.1, random_state=4000)

regularizer_coef=0.0000001/1024   


Number_of_pilot=36
epochs=50

network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=600,
                                 Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef ,on_cloud=on_cloud)

network.train(X_train,X_train, epochs=epochs, log_path=log_path)

Image_filename=log_path+"/generated.png"
Test_Error,Y_all,X_all=network.generateAndPlot(X_test,50,fileName=Image_filename)
print(np.sqrt(Test_Error))

print(np.mean(np.sqrt(Test_Error)))



fig = plt.figure()
W=network.encoder.layers[1].get_weights()
W2=W[0]#np.sum(np.abs(W[0]),axis=1)/W[0].shape[1]
plt.imshow(W2.reshape(X_train[0].shape))

if on_cloud==1:
	fig.savefig(log_path+"/W.png")
else:
	plt.show()


fig2 = plt.figure()
plt.hist(W2[0])
fig2.savefig(log_path+"/W_hist.png")
if on_cloud==1:
	fig2.savefig(log_path+"/W_hist.png")
else:
	plt.show()


