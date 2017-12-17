
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
import math

#matplotlib.use('Qt5Agg')


flags = tf.app.flags
flags.DEFINE_string("dataset", "", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("logdir", "", "Directory name to save the image samples [output]")
FLAGS = flags.FLAGS


# In[7]:
#def main(_):
on_cloud=1
normalize_mode=1 # 1: (a+5)/10, #2: MinMaxScaler, 3: noting 

regularizer_coef=0.00000006/1024   #16 (10)
#regularizer_coef=0.0000001/1024   #7 (36) #14 (8)
regularizer_coef=0.0000002/1024   #9 (36) #12 (16) #39
regularizer_coef=0.00000001/1024   #81 
regularizer_coef=0.000000046/1024   #82
#regularizer_coef=0.0000001/1024   #82


# 56 --> 24, 4SNR
# 57 --> 48, 4SNR
# 58 --> 48, Vari SNR
# 60 --> 36, 4SNR

#63 -->  9 and 12
#64 34--> 18 only
#67 --> no_noise-48 pilot - Mode 4
#68 --> no_noise-48 pilot - Mode 2
#68 --> no_noise-48 pilot - Mode 1

encoded_dim=500
epochs=40

Number_of_pilot=48
SNR_H=3
SNR_L=12
if normalize_mode==4:
  Noise_var_L=pow(10,(-SNR_H/10))/25
  Noise_var_H=pow(10,(-SNR_L/10))/25
elif normalize_mode==1:
  Noise_var_L=pow(10,(-SNR_H/10))/100
  Noise_var_H=pow(10,(-SNR_L/10))/100
else:
  Noise_var_L=pow(10,(-SNR_H/10))
  Noise_var_H=pow(10,(-SNR_L/10))

print(Noise_var_H)
print(Noise_var_L)
print(np.log10(Noise_var_L*100)+1.5)
print(np.log10(Noise_var_H*100)+1.5)

if (on_cloud == 1):
    log_path = os.path.join("/output/",FLAGS.logdir)
    #data_path = os.path.join("/data/",FLAGS.dataset)
    data_path = os.path.join(FLAGS.dataset)
else:
    log_path = os.path.join("./output/",FLAGS.logdir)
    #data_path = os.path.join(FLAGS.dataset)
    data_path = os.path.join("/Local_data/chimage_data_3/","")

Data_file=data_path+"/Ch_real_VehA_14.mat"
#Data_file="Ch_real_VehA_14.mat"


channels = scipy.io.loadmat(Data_file)['channels']
reals = np.real(channels)
imags = np.imag(channels)
all_channel_images = np.vstack([reals, imags])

if normalize_mode==1:
    all_channel_images = (all_channel_images+5)/10.0
elif normalize_mode==4:
    all_channel_images = (all_channel_images)/5

#print(np.amax(all_channel_images))

#print(np.amin(all_channel_images))

#exit()

X_train , X_test = train_test_split(all_channel_images, test_size=.05, random_state=4000)



network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=encoded_dim,
                                 Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                 on_cloud=on_cloud,test_mode =0 , log_path=log_path, normalize_mode=normalize_mode,
                                 Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H)

network.train(X_train, epochs=epochs)


Test_network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=encoded_dim,
                                      Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                      on_cloud=on_cloud,test_mode =1 , log_path=log_path, normalize_mode=normalize_mode,
                                      Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H)

Image_filename=log_path+"/generated.png"
Test_Error,Y_all,X_all=Test_network.generateAndPlot(X_test,50,fileName=Image_filename)
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


