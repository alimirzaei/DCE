
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

#regularizer_coef=0.0000002/1024   
Train_model=1
Test_model=1


#126
regularizer_coef=0.0000000005   
encoded_dim=60
Number_of_pilot=48

#128
regularizer_coef=0.000000001      
encoded_dim=40
Number_of_pilot=36

#131
regularizer_coef=0.000000001      
encoded_dim=40
Number_of_pilot=48

#139
regularizer_coef=0.0000000001      
encoded_dim=25
Number_of_pilot=48


#140
regularizer_coef=0.0000000001      
encoded_dim=40
Number_of_pilot=36

#142 5-16
regularizer_coef=0.0000000001      
encoded_dim=30
Number_of_pilot=48

#143 12-14
regularizer_coef=0.0000000001      
encoded_dim=10
Number_of_pilot=48

#144 12-14
regularizer_coef=0.0000000001      
encoded_dim=600
Number_of_pilot=48

#146 12-14
regularizer_coef=0.0000000001      
encoded_dim=40
Number_of_pilot=48


#encoded_dim=40
epochs=40

SNR_H=12
SNR_L=14


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
print((np.log10(Noise_var_L*100)+2)/2)
print((np.log10(Noise_var_H*100)+2)/2)



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

if Train_model==1:
  network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=encoded_dim,
                                  Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                  on_cloud=on_cloud,test_mode =0 , log_path=log_path, normalize_mode=normalize_mode,
                                  Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H)

  network.train(X_train, epochs=epochs)

if Test_model==1:
  Test_network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=encoded_dim,
                                        Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                        on_cloud=on_cloud,test_mode =1 , log_path=log_path, normalize_mode=normalize_mode,
                                        Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H)

  Image_filename=log_path+"/generated.png"
  Test_Error,Y_all,X_all=Test_network.generateAndPlot(X_test,50,fileName=Image_filename)
  print(np.sqrt(Test_Error))

  print(np.mean(np.sqrt(Test_Error)))



  fig = plt.figure()
  W=Test_network.selector.layers[1].get_weights()
  W2=W[0]#np.sum(np.abs(W[0]),axis=1)/W[0].shape[1]
  plt.imshow(W2.reshape(X_train[0].shape))

  if on_cloud==1:
  	fig.savefig(log_path+"/W.png")
  else:
    fig.savefig(log_path+"/W.png")
    plt.show()


  fig2 = plt.figure()
  plt.hist(W2[0])
  if on_cloud==1:
  	fig2.savefig(log_path+"/W_hist.png")
  else:
    fig2.savefig(log_path+"/W_hist.png")
    plt.show()


