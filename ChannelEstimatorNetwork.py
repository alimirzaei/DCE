# -*- coding: utf-8 -*-


import copy
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, Lambda, concatenate, BatchNormalization
from keras import regularizers
from keras.initializers import RandomNormal

from keras.optimizers import Adam
import numpy as np
from keras import backend as K
from CustomLayers import MaskLayer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def AddNoise(input_args):
    x = input_args[0]
    noise = input_args[1]
    return x+noise

class SparseEstimatorNetwork():
    def __init__(self, img_shape=(28, 28), encoded_dim=2, Number_of_pilot=30,
                 regularizer_coef=1e-6 ,on_cloud=1, test_mode=0, log_path='.', normalize_mode=2, Noise_var_L=.01, Noise_var_H=.1):
        self.encoded_dim = encoded_dim
        self.optimizer = Adam(0.00005)
        self.img_shape = img_shape
        self.Number_of_pilot=Number_of_pilot
        self.regularizer_coef=regularizer_coef
        self.test_mode=test_mode        
        self.log_path=log_path        
        self.on_cloud=on_cloud
        self.normalize_mode=normalize_mode
        self.Noise_var_L=Noise_var_L
        self.Noise_var_H=Noise_var_H
        if self.normalize_mode==2:
            self.scaler = MinMaxScaler((-1,1))
        self._initAndCompileFullModel(img_shape, encoded_dim)
        #self.scaler = StandardScaler(with_mean=True, with_std=True)
        
        

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
        encoder.add(Dense(encoded_dim, activation='sigmoid'))
        #encoder.add(BatchNormalization())
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
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim+1))
        #decoder.add(Dense(1000, activation='relu', kernel_regularizer= regularizers.l1(0.00000002/1024)))
        decoder.add(Dense(1000, activation='relu')) 
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
        noise = Input(shape=img_shape)
        variance = Input(shape=(1,))
        noisy_image = Lambda(AddNoise)([img, noise])
     
        #concated = Concatenate([Flatten(input_shape=img_shape)(noisy_image), variance])
        encoded_repr = self.encoder(noisy_image)
        concated = concatenate([encoded_repr, variance])
        gen_img = self.decoder(concated)
        self.autoencoder = Model([img, noise, variance], gen_img)
        #self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
        self.autoencoder.compile(optimizer=self.optimizer, loss='mae')
        if self.test_mode==1:
            if self.on_cloud==0:
                Weigth_data=self.log_path+"/"+"weights.hdf5"
                if (os.path.isfile(Weigth_data)):
                    self.autoencoder.load_weights(Weigth_data)
                else:
                    print("train the model first!!!")
                if self.normalize_mode==2:
                    scaler_filename = self.log_path+"/scaler.save"
                    if (os.path.isfile(scaler_filename)):
                        print("loaded scaleer")
                        self.scaler = joblib.load(scaler_filename)
                    else:
                        print("train the model first!!!")
            else:
                #might be changed if the weights location changes
                Weigth_data=self.log_path+"/"+"weights.hdf5"
                if (os.path.isfile(Weigth_data)):
                    self.autoencoder.load_weights(Weigth_data)
                else:
                    print("train the model first!!!")
                if self.normalize_mode==2:
                    scaler_filename = self.log_path+"/scaler.save"
                    if (os.path.isfile(scaler_filename)):
                        self.scaler = joblib.load(scaler_filename)
                    else:
                        print("train the model first!!!")


    def train(self, x_in, batch_size=32, epochs=5):

        Num_noise_per_image=4

        x_in= np.tile(x_in, (Num_noise_per_image,1,1))

        if self.normalize_mode==2:
            x_scaled = self.scaler.fit_transform(x_in.reshape(len(x_in),-1))
        else:
            x_scaled=x_in

        x_scaled_reshped =  x_scaled.reshape(x_in.shape)
        if (os.path.isfile(self.log_path+'/weights.hdf5')):
            self.autoencoder.load_weights('weights.hdf5')
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

        noises = []
        variances = []

        for i in range(len(x_in)):
            var = np.random.uniform(self.Noise_var_L, self.Noise_var_H)
            noise = np.sqrt(var)/np.sqrt(2)*np.random.randn(*x_in[0].shape)
            if self.normalize_mode==4:
                variances.append(25*var)
            elif self.normalize_mode==1:
                noise=noise
                variances.append(np.log10(100*var)+1.5)
                #variances.append(0)
            else:
                variances.append(var)
            noises.append(noise)

        # for i in range(0,len(x_in),Num_noise_per_image):
        #     var = pow(10,(-3/10))/25
        #     noise = np.sqrt(var)/np.sqrt(2)*np.random.randn(*x_in[0].shape)
        #     if self.normalize_mode==4:
        #         variances.append(25*var)
        #     elif self.normalize_mode==1:
        #         noise=noise
        #         variances.append(2*np.log10(100*var)+1)
        #     else:
        #         variances.append(var)
        #     noises.append(noise)
        #     var = pow(10,(-6/10))/25
        #     noise = np.sqrt(var)/np.sqrt(2)*np.random.randn(*x_in[0].shape)
        #     if self.normalize_mode==4:
        #         variances.append(25*var)
        #     elif self.normalize_mode==1:
        #         noise=noise
        #         variances.append(2*np.log10(100*var)+1)
        #     else:
        #         variances.append(var)
        #     noises.append(noise)
        #     var = pow(10,(-9/10))/25
        #     noise = np.sqrt(var)/np.sqrt(2)*np.random.randn(*x_in[0].shape)
        #     if self.normalize_mode==4:
        #         variances.append(25*var)
        #     elif self.normalize_mode==1:
        #         noise=noise
        #         variances.append(2*np.log10(100*var)+1)
        #     else:
        #         variances.append(var)
        #     noises.append(noise)
        #     var = pow(10,(-12/10))/25
        #     noise = np.sqrt(var)/np.sqrt(2)*np.random.randn(*x_in[0].shape)
        #     if self.normalize_mode==4:
        #         variances.append(25*var)
        #     elif self.normalize_mode==1:
        #         noise=noise
        #         variances.append(2*np.log10(100*var)+1)
        #     else:
        #         variances.append(var)
        #     noises.append(noise)
        #     # var = pow(10,(-9/10))/25
        #     # noise = np.sqrt(var)/np.sqrt(2)*np.random.randn(*x_in[0].shape)
        #     # if self.normalize_mode==4:
        #     #     variances.append(25*var)
        #     # elif self.normalize_mode==1:
        #     #     noise=noise
        #     #     variances.append(2*np.log10(100*var)+1)
        #     # else:
        #     #     variances.append(var)
        #     # noises.append(noise)

        noises = np.array(noises)
        variances = np.array(variances)

        self.autoencoder.fit([x_scaled_reshped, noises, variances], x_scaled_reshped, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.15,
                              callbacks=[earlyStopping,
                                        keras.callbacks.ModelCheckpoint(self.log_path+"/"+'weights.hdf5', 
                                           verbose=0, 
                                           monitor='val_loss',
                                           #save_best_only=False, 
                                           save_best_only=True, 
                                           save_weights_only=False, 
                                           mode='auto', 
                                           period=1)
                                        #,keras.callbacks.TensorBoard(log_dir=self.log_path, 
                                        #            histogram_freq=0, 
                                        #            batch_size=batch_size, write_graph=True,
                                        #            write_grads=False,
                                        #            write_images=False,
                                        #            embeddings_freq=0,
                                        #            embeddings_layer_names=None,
                                        #            embeddings_metadata=None)
                                        ])
        if self.normalize_mode==2:
            scaler_filename = "/scaler.save"
            joblib.dump(self.scaler, self.log_path+scaler_filename)         

    def test(self, x_in, var):
        if self.normalize_mode==2:
            x_scaled = self.scaler.transform(x_in.reshape(len(x_in),-1))
        else:
            x_scaled=x_in

        x_scaled_reshped =  x_scaled.reshape(x_in.shape)
        y = self.autoencoder.predict([x_scaled_reshped.reshape(*x_in.shape),np.zeros(x_in.shape), var*np.ones((len(x_in), 1))])
        if self.normalize_mode==2:
            y_true = self.scaler.inverse_transform(y.reshape(len(y),-1))
        else:
            y_true=y

        return y_true.reshape(*x_in.shape)
        

    def FindEstiamte(self, x_test, fileName="test.png"):
        #fig = plt.figure(figsize=[20, 20/3])
        x_in = x_test
        y = self.test(x_in.reshape(1,x_in.shape[0],x_in.shape[1]))

        fig = plt.figure(figsize=[20, 20/2])
        i=0
        ax = fig.add_subplot(1, 2, i*2+1)
        ax.set_axis_off()
        ax.imshow(x_in)
        ax = fig.add_subplot(1, 2, i*2+2)
        ax.set_axis_off()
        ax.imshow(y[0]) #Layer cut
        fig.savefig(fileName)
        return y

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
            y = self.test(x.reshape(1,x_test.shape[1],x_test.shape[2]),0)
            ax = fig.add_subplot(n, 3, i*3+1)
            ax.set_axis_off()
            ax.imshow(x)
            ax = fig.add_subplot(n, 3, i*3+2)
            ax.set_axis_off()
            ax.imshow(y[0]) #Layer cut
            
            Sampled_image = Sampled_image_model([x.reshape(1,x_test.shape[1],x_test.shape[2])])[0]
            #Sampled_image[Sampled_image<1e-6]=0
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
    network = SparseEstimatorNetwork(encoded_dim=10)
    network.train(x_train, epochs=1, a=.1, b=1)
    #y = network.test(x_test[0:10])
    network.generateAndPlot(x_test)
