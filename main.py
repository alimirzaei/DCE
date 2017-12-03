
# coding: utf-8

# In[1]:

from ChannelEstimatorNetwork import SparseEstimatorNetwork
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Qt5Agg')

channels = scipy.io.loadmat('Ch_real_VehA_14.mat')['channels']
reals = np.real(channels)
imags = np.imag(channels)
all_channel_images = np.vstack([reals, imags])


X_train , X_test = train_test_split(all_channel_images)

regularizer_coef=0.000002/1024   #52  (36 POINTS) 50 iter early #53 Large pics!

Number_of_pilot=36
epochs=1

network = SparseEstimatorNetwork(img_shape=X_train[0].shape, encoded_dim=600,
                                 Number_of_pilot=Number_of_pilot)

network.train(X_train,X_train, epochs=epochs)


Test_Error,Y_all,X_all=network.generateAndPlot(X_test,50)
print(np.sqrt(Test_Error))

print(np.mean(np.sqrt(Test_Error)))



fig = plt.figure()
W=network.encoder.layers[1].get_weights()
W2=W[0]#np.sum(np.abs(W[0]),axis=1)/W[0].shape[1]
plt.imshow(W2.reshape(X_train[0].shape))
plt.show()


#fig2 = plt.figure()
#plt.hist(W2[0])
#fig2.savefig(log_path+"/W_hist.png")
# plt.show()


