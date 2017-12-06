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
Number_of_pilot=16

log_path='../Share_weights'



network = SparseEstimatorNetwork(img_shape=input_shape, encoded_dim=encoded_dim, Number_of_pilot=Number_of_pilot,
                                      regularizer_coef=regularizer_coef ,on_cloud=on_cloud,test_mode =1 , 
                                      log_path=log_path)


@app.route("/estimate_channel", methods = ['POST'])
def estimate_channel():
	global tmp
	global network
	channel = np.array(json.loads(request.data.decode('utf-8')))

	channelscale= (channel+5)/10.0
	output=network.FindEstiamte(channelscale,fileName='internal_test.jpg')
	output= output*10-5

	result = json.dumps(output[0].tolist())
	return result

if __name__ == "__main__":
	app.run()