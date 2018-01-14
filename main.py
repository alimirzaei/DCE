
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
flags.DEFINE_string("dataset2", "", "The name of the second dataset [celebA, mnist, lsun]")
FLAGS = flags.FLAGS

data_type=2 # 0: 40K channel, 1: 40K channel and noisy channel at 12db

# In[7]:
#def main(_):

on_cloud=0
if (on_cloud == 1):
    log_path = os.path.join("/output/",FLAGS.logdir)
    #data_path = os.path.join("/data/",FLAGS.dataset)
    data_path = os.path.join(FLAGS.dataset)
    data_path2 = os.path.join(FLAGS.dataset2)
else:
    log_path = os.path.join("./output/",FLAGS.logdir)
    #data_path = os.path.join(FLAGS.dataset)
    if data_type==0:
      data_path = os.path.join("/Local_data/chimage_data_3/","")
      data_path = os.path.join("/Dropbox/Working_dir/Tensorflow_home/Share_weights/Perfect_channel","")
    elif data_type==1:
      data_path = os.path.join("/Dropbox/Working_dir/Tensorflow_home/Share_weights/Perfect_channel/","")
      data_path2 = os.path.join("/Dropbox/Working_dir/Tensorflow_home/Share_weights/Noisy_channels/","")
    elif data_type==2:
      data_path = os.path.join("/Dropbox/Working_dir/Tensorflow_home/Share_folder/perfect_h21/","")
      data_path2 = os.path.join("/Dropbox/Working_dir/Tensorflow_home/Share_folder/Noisy_H22/","")

if data_type==0:
  #Data_file=data_path+"/Ch_real_VehA_14.mat"
  Data_file=data_path+"/My_perfect_H_12.mat"
elif data_type==1:
  Data_file1=data_path+"/My_perfect_H_12.mat"
  Data_file2=data_path2+"/My_noisy_H_12.mat"
elif data_type==2:
  Data_file1=data_path+"/My_perfect_H_22.mat"
  Data_file2=data_path2+"/My_noisy_H_22.mat"


#regularizer_coef=0.0000002/1024   
Train_model=1
Test_model=1
Enable_conv=1
Fixed_pilot=1
Enable_auto=0
normalize_mode=2 # 1: (a+5)/10, #2: MinMaxScaler, 3: noting 
Drou_out_sel=0

SNR_H=22
SNR_L=22

regularizer_coef=0.0000000001      
encoded_dim=400

#Number_of_pilot=4

Number_of_pilot=48




#encoded_dim=40
epochs=8


if normalize_mode==4:
  Noise_var_L=pow(10,(-SNR_H/10))/25
  Noise_var_H=pow(10,(-SNR_L/10))/25
elif normalize_mode==1:
  Noise_var_L=pow(10,(-SNR_H/10))/100
  Noise_var_H=pow(10,(-SNR_L/10))/100
elif normalize_mode==5:
  Noise_var_L=pow(10,(-SNR_H/10))
  Noise_var_H=pow(10,(-SNR_L/10))
else:
  Noise_var_L=pow(10,(-SNR_H/10))
  Noise_var_H=pow(10,(-SNR_L/10))

print(Noise_var_H)
print(Noise_var_L)

if Noise_var_L==Noise_var_H:
    print(.1)
else:
    print((-np.log10(Noise_var_H)+np.log10(Noise_var_L))/(-np.log10(Noise_var_H)+np.log10(Noise_var_L)))



if data_type==0:
  #channels = scipy.io.loadmat(Data_file)['channels']
  channels = scipy.io.loadmat(Data_file)['My_perfect_H']

  reals = np.real(channels)
  imags = np.imag(channels)
  all_channel_images = np.vstack([reals, imags])

  if normalize_mode==1:
    all_channel_images = (all_channel_images+5)/10.0
  if normalize_mode==5:
    all_channel_images = (all_channel_images+5)
  elif normalize_mode==4:
    all_channel_images = (all_channel_images)/5

  X_train , X_test = train_test_split(all_channel_images, test_size=.05, random_state=4000)

elif data_type==1 or data_type==2:

  channels_noisy = scipy.io.loadmat(Data_file2)['My_noisy_H']
  reals = np.real(channels_noisy)
  imags = np.imag(channels_noisy)
  all_channel_noisy_images = np.vstack([reals, imags])
  
  channels_perfect = scipy.io.loadmat(Data_file1)['My_perfect_H']
  reals = np.real(channels_perfect)
  imags = np.imag(channels_perfect)
  all_channel_perfect_images = np.vstack([reals, imags])

  if normalize_mode==1:
    all_channel_noisy_images = (all_channel_noisy_images+5)/10.0
    all_channel_perfect_images = (all_channel_perfect_images+5)/10.0
  if normalize_mode==5:
    all_channel_noisy_images = (all_channel_noisy_images+5)
    all_channel_perfect_images = (all_channel_perfect_images+5)

  X_train , X_test, Y_train, Y_test = train_test_split(all_channel_noisy_images,all_channel_perfect_images, test_size=.05, random_state=4000)



if Train_model==1:
  network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=encoded_dim,
                                Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                on_cloud=on_cloud,test_mode =0 , log_path=log_path, normalize_mode=normalize_mode,
                                Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H, data_type=data_type,
                                Enable_conv=Enable_conv,Fixed_pilot=Fixed_pilot,Enable_auto=Enable_auto,Drou_out_sel=Drou_out_sel)

  if data_type==0:
    network.train(X_train, epochs=epochs)
  elif data_type==1 or data_type==2:
    network.train(X_train, Y_train, epochs=epochs)



if Test_model==1:
  Test_network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=encoded_dim,
                                        Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                        on_cloud=on_cloud,test_mode =1 , log_path=log_path, normalize_mode=normalize_mode,
                                        Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H, data_type=data_type,
                                        Enable_conv=Enable_conv,Fixed_pilot=Fixed_pilot,Enable_auto=Enable_auto,Drou_out_sel=Drou_out_sel)

  Image_filename=log_path+"/generated.png"
  if data_type==0:
    Test_Error,Test_error_int,Y_all,X_all=Test_network.generateAndPlot(X_test,n=50,fileName=Image_filename)
  elif data_type==1 or data_type==2:
    Test_Error,Test_error_int,Y_all,X_all=Test_network.generateAndPlot(X_test,Y_test,n=50,fileName=Image_filename)

  print((Test_Error))

  print(np.mean((Test_Error)))


  print("------------------")
  print((Test_error_int))

  print(np.mean((Test_error_int)))


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


