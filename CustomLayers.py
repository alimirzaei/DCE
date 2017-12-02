# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 01:52:24 2017

@author: Ali
"""

from keras import regularizers
from keras import activations
from keras import initializers
from keras import constraints
from keras.engine import InputSpec
from keras.engine import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np

class MaskLayer(Layer):
    """ 
    """

    def __init__(self,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 Number_of_pilot=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MaskLayer, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.Number_of_pilot=Number_of_pilot
        
    def my_add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   trainable=True):

        initializer = initializers.get(self.kernel_initializer)
        if dtype is None:
            dtype = K.floatx()

        select_vec=K.variable(initializer(shape),
                            dtype=dtype,
                            name=name,
                            constraint=self.kernel_constraint)
        weight=select_vec
        
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(select_vec))
        
        if trainable:
            self._trainable_weights.append(select_vec)
        else:
            self._non_trainable_weights.append(select_vec)
        
        return weight


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim=input_dim;

        self.kernel = self.my_add_weight(shape=(1, input_dim),
                                      name='kernel')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        if self.Number_of_pilot==None:
            Num_pilot=100
        else:
            Num_pilot=self.Number_of_pilot
            
        select_vec_abs=self.kernel
        maxes, indx= tf.nn.top_k((select_vec_abs),Num_pilot)
        self.treshold=maxes[0,Num_pilot-1]

        Zeros_tf=tf.constant(np.zeros(shape=self.kernel.shape), dtype=tf.float32)
        Ones_tf=tf.constant(np.ones(shape=self.kernel.shape), dtype=tf.float32)
 
        mask=tf.where (select_vec_abs>self.treshold,x=Ones_tf, y=Zeros_tf)
        output=inputs*mask
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MaskLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
