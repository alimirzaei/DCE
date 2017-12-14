from flask import Flask
from flask import request
import json
import os
import numpy as np
from ChannelEstimatorNetwork import SparseEstimatorNetwork
app = Flask(__name__)

 
on_cloud=0
input_shape=(72, 14)
encoded_dim=200   

regularizer_coef=0.0000001/1024
Number_of_pilot=36

regularizer_coef=0.0000002/1024   
normalize_mode=3  # 1: (a+5)/10, #2: MinMaxScaler, 3: noting 

regularizer_coef=0.000000002/1024   #40 
normalize_mode=2  # 1: (a+5)/10, #2: MinMaxScaler, 3: noting 

regularizer_coef=0.00000001/1024   #40 
normalize_mode=4  # 1: (a+5)/10, #2: MinMaxScaler, 3: noting 


Number_of_pilot=48
SNR_H=9
SNR_L=0
Noise_var_L=pow(10,(-SNR_H/10))/25
Noise_var_H=pow(10,(-SNR_L/10))/25


log_path='../Share_weights'


Test_network = SparseEstimatorNetwork(img_shape=input_shape, encoded_dim=encoded_dim,
                                      Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                      on_cloud=on_cloud,test_mode =1 , log_path=log_path, normalize_mode=normalize_mode,
                                      Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H)


import scipy.io
import io
import base64
@app.route("/estimate_channel", methods = ['POST'])
def estimate_channel():
	#global tmp
	#global network
	#channel = np.array(json.loads(request.data.decode('utf-8')))
   # s = request.data.split('*')
   # image_str, var_str = s[0], s[1]
   # image = np.array(np.matrix(image_str))
   # var = float(var_str)
   # y = network.test(image, var)
	#channelscale= (channel+5)/10.0
	#output=network.FindEstiamte(channelscale,fileName='internal_test.jpg')
	#output= output*10-5
	#result = json.dumps(output[0].tolist())
	#return result
    #print(request.data)
    #print(base64.b64decode(request.data))
    b = scipy.io.loadmat(io.BytesIO(base64.b64decode(request.data)))
    out = b
    image = b['cel'][0][0]
    var = b['cel'][0][1][0][0]
    if normalize_mode==1:
    	image= (image+5)/10.0
    elif normalize_mode==4:
    	image= (image)/5

    y = Test_network.test(image, var)
    
    if normalize_mode==1:
    	y= y*10-5
    elif normalize_mode==4:
    	y=y*5

    output = io.BytesIO()
    o ={}
    o['out']=y[0]
    scipy.io.savemat(output, o)
    return base64.b64encode(output.getvalue())

if __name__ == "__main__":
	app.run()