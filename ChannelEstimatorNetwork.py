# -*- coding: utf-8 -*-


import copy
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape
from keras import regularizers
from keras.initializers import RandomNormal

from keras.optimizers import Adam
import numpy as np
from keras import backend as K
from CustomLayers import MaskLayer


import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class SparseEstimatorNetwork():
    def __init__(self, img_shape=(28, 28), encoded_dim=2, Number_of_pilot=30,
                 regularizer_coef=1e-6 ,on_cloud=1):
        self.encoded_dim = encoded_dim
        self.optimizer = Adam(0.0001)
        self.optimizer_discriminator = Adam(0.00001)
        self.img_shape = img_shape
        self.Number_of_pilot=Number_of_pilot
        self.regularizer_coef=regularizer_coef
        self.on_cloud=on_cloud
        self._initAndCompileFullModel(img_shape, encoded_dim)


    def _genEncoderModel(self, img_shape, encoded_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
                """

        encoder = Sequential()
        encoder.add(Flatten(input_shape=img_shape))
        encoder.add(MaskLayer(input_dim=img_shape, activation='linear', 
                                    kernel_regularizer= regularizers.l1(self.regularizer_coef), 
                                    #kernel_regularizer= My_l1_reg, 
                                    Number_of_pilot=self.Number_of_pilot))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(encoded_dim))
        encoder.summary()
        return encoder

    def _getDecoderModel(self, encoded_dim, img_shape):
        """ Build Decoder Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
            img_shape (tuple) : shape of target images
        Return:
            A sequential keras model
        """
        decoder = Sequential()
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim))
        decoder.add(Dense(1000, activation='relu'))
        #decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        decoder.summary()
        return decoder


    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        img = Input(shape=img_shape)
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        self.autoencoder.compile(optimizer=self.optimizer, loss='mse')


    def train(self, x_in, x_out, batch_size=32, epochs=5, log_path='.'):
        if (os.path.isfile(log_path+'/weights.hdf5')):
            self.autoencoder.load_weights('weights.hdf5')
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        self.autoencoder.fit(x_in, x_out, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.15,
                              callbacks=[earlyStopping,
                                        keras.callbacks.ModelCheckpoint(log_path+"/"+'weights.hdf5', 
                                           verbose=0, 
                                           monitor='val_loss',
                                           #save_best_only=False, 
                                           save_best_only=True, 
                                           save_weights_only=False, 
                                           mode='auto', 
                                           period=1)
                                        #,keras.callbacks.TensorBoard(log_dir=log_path, 
                                        #            histogram_freq=0, 
                                        #            batch_size=batch_size, write_graph=True,
                                        #            write_grads=False,
                                        #            write_images=False,
                                        #            embeddings_freq=0,
                                        #            embeddings_layer_names=None,
                                        #            embeddings_metadata=None)
                                        ])

    def generateAndPlot(self, x_test, n = 10, fileName="generated.png"):
        Sampled_image_model = K.function([self.encoder.layers[0].input],
                                  [self.encoder.layers[1].output])

        fig = plt.figure(figsize=[20, 20*n/3])
        Test_error=np.array(np.zeros(shape=(1,n)))
        Y_all=[]
        X_all=[]
        for i in range(n):
            x_in = x_test[np.random.randint(len(x_test))]
            x=copy.copy(x_in)
            y = self.autoencoder.predict(x.reshape(1,x_test.shape[1],x_test.shape[2]))
            ax = fig.add_subplot(n, 3, i*3+1)
            ax.set_axis_off()
            ax.imshow(x)
            ax = fig.add_subplot(n, 3, i*3+2)
            ax.set_axis_off()
            ax.imshow(y[0]) #Layer cut
            
            Sampled_image = Sampled_image_model([x.reshape(1,x_test.shape[1],x_test.shape[2])])[0]
            Sampled_image[Sampled_image<1e-6]=0
            ax = fig.add_subplot(n, 3, i*3+3)
            ax.set_axis_off()
            ax.imshow(Sampled_image.reshape(x_test.shape[1],x_test.shape[2]))
            Test_error[0,i]=np.mean(np.square(x-y[0]))
            X_all.append(x)
            Y_all.append(y[0])

        fig.savefig(fileName)
        return Test_error, Y_all, X_all
        
if __name__=='__main__':
    # here is to just test the network
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    network = SparseEstimatorNetwork()
    network.train(x_train, x_train)
    
    
