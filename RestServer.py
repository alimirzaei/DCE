from flask import Flask
from flask import request
import json
import os
import numpy as np
from ChannelEstimatorNetwork import SparseEstimatorNetwork
app = Flask(__name__)

 
on_cloud=0
input_shape=(72, 14)


#126
regularizer_coef=0.0000000005   
encoded_dim=60
Number_of_pilot=48
log_path='../Share_weights/48_60_p0000000005_126'

#128
regularizer_coef=0.000000001      
encoded_dim=40
Number_of_pilot=36
log_path='../Share_weights/36_40_p000000001_128'

#130
regularizer_coef=0.0000002/1024      
encoded_dim=40
Number_of_pilot=36
log_path='../Share_weights/36_40_p0000000005_130_weigh_sel'


#131
regularizer_coef=0.000000001      
encoded_dim=40
Number_of_pilot=48
log_path='../Share_weights/48_40_p000000001_131'

#135
regularizer_coef=0.0000000001      
encoded_dim=200
Number_of_pilot=48
log_path='../Share_weights/48_200_p000000001_135'


#136
regularizer_coef=0.0000000001      
encoded_dim=250
Number_of_pilot=48
log_path='../Share_weights/48_250_p000000001_136'


#139
regularizer_coef=0.0000000001      
encoded_dim=25
Number_of_pilot=48
log_path='../Share_weights/48_25_Fixed_139_only12'



#140
regularizer_coef=0.0000000001      
encoded_dim=40
Number_of_pilot=36
log_path='../Share_weights/48_40_Fixed_140_only12'

# #138
# regularizer_coef=0.0000000001      
# encoded_dim=250
# Number_of_pilot=48
# log_path='../Share_weights/48_250_p000000001_138_only12'

#142 5-16
regularizer_coef=0.0000000001      
encoded_dim=30
Number_of_pilot=48
log_path='../Share_weights/48_30_Fixed_142'


#143 12-14
regularizer_coef=0.0000000001      
encoded_dim=10
Number_of_pilot=48
log_path='../Share_weights/48_10_Fixed_143_12and13'

#144 12-14
regularizer_coef=0.0000000001      
encoded_dim=600
Number_of_pilot=48
log_path='../Share_weights/48_600_Fixed_143_12and13'

#146 12-14
regularizer_coef=0.0000000001      
encoded_dim=40
Number_of_pilot=48
log_path='../Share_weights/48_600_Fixed_143_12and13_newmod'

#148 12-12
regularizer_coef=0.0000000001      
encoded_dim=40
Number_of_pilot=48
log_path='../Share_weights/48_Noisy_chan'
data_type=1

#149 12-12
regularizer_coef=0.0000000001      
encoded_dim=40
Number_of_pilot=48
log_path='../Share_weights/48_conv'

#149 12-12
regularizer_coef=0.0000000001      
encoded_dim=300
Number_of_pilot=48
log_path='../Share_weights/48_conv_drop_noisy'

#154 12-12
regularizer_coef=0.0000000001      
encoded_dim=100
Number_of_pilot=48
log_path='../Share_weights/48_conv_drop'

#163 3-15 - 48
regularizer_coef=0.0000000001      
encoded_dim=300
Number_of_pilot=48
log_path='../Share_weights/48_conv_drop_v2'

#local 3-15 - 34
regularizer_coef=0.0000000001      
encoded_dim=300
Number_of_pilot=32
log_path='../Share_weights/32_conv_partial'

#164 3-15 - 34
regularizer_coef=0.0000000001      
encoded_dim=300
Number_of_pilot=32
log_path='../Share_weights/32_conv'

regularizer_coef=0.0000000001      
encoded_dim=300
Number_of_pilot=48
log_path='../Share_weights/48_no_conv_12'

regularizer_coef=0.0000000001      
encoded_dim=300
Number_of_pilot=48
log_path='../Share_weights/48_conv_12'

regularizer_coef=0.0000000001      
encoded_dim=250
Number_of_pilot=48
log_path='../Share_weights/48_conv_12_newnorm'
log_path='../Share_weights/48_conv_12_newnorm_2'
log_path='../Share_weights/48_conv_3_12_newnorm_newpilot'
log_path='../Share_weights/48_new_struct'
log_path='../Share_weights/48_new_struct_type2'

Enable_auto=1


Number_of_pilot=36
log_path='../Share_weights/36_flex_new_struct'
Fixed_pilot=0

Number_of_pilot=36
log_path='../Share_weights/36_fixed'
Fixed_pilot=1
SNR_H=12
SNR_L=12
data_type=0  #??

Number_of_pilot=36
log_path='../Share_weights/36_fixed_2_16'
Fixed_pilot=1
SNR_H=2
SNR_L=16
data_type=0  #??

Number_of_pilot=36
log_path='../Share_weights/36_fixed_14'
Fixed_pilot=1
SNR_H=14
SNR_L=14
data_type=0  #??
Enable_auto=0


Number_of_pilot=36
log_path='../Share_weights/36_fixed_22'
Fixed_pilot=1
SNR_H=22
SNR_L=22
data_type=0  #??

Number_of_pilot=48
log_path='../Share_weights/36_fixed_22'
Fixed_pilot=1
SNR_H=22
SNR_L=22
data_type=0  #??


Number_of_pilot=48
log_path='../Share_weights/48_fixed_22_noAuto'
Fixed_pilot=1
SNR_H=22
SNR_L=22
data_type=0  #??
Enable_auto=0


Number_of_pilot=48
log_path='../Share_weights/48_fixed_12_noAuto_type1'
Fixed_pilot=1
SNR_H=12
SNR_L=12
data_type=1  #??
Enable_auto=0

Number_of_pilot=36
log_path='../Share_weights/36_fixed_12_noAuto_type1'
Fixed_pilot=1
SNR_H=12
SNR_L=12
data_type=1  #??
Enable_auto=0

Number_of_pilot=36
log_path='../Share_weights/36_fixed_22_noAuto_type2'
Fixed_pilot=1
SNR_H=22
SNR_L=22
data_type=2  #??
Enable_auto=0
Drou_out_sel=0

Number_of_pilot=36
log_path='../Share_weights/36_fixed_22_noAuto_type2_newStruct_mae'
Fixed_pilot=1
SNR_H=22
SNR_L=22
data_type=2  #??
Enable_auto=0
Drou_out_sel=0


Number_of_pilot=48
log_path='../Share_weights/48_fixed_22_noAuto_type2_newStruct2_mae'
Fixed_pilot=1
SNR_H=22
SNR_L=22
data_type=2  #??
Enable_auto=0
Drou_out_sel=0
Enable_conv=1

Number_of_pilot=48
encoded_dim=400

log_path='../Share_weights/48_fixed_22_noAuto_type2_newStruct2_mse'
Fixed_pilot=1
SNR_H=22
SNR_L=22
data_type=2  #??
Enable_auto=0
Drou_out_sel=0
Enable_conv=1


Train_model=0
Test_model=1
normalize_mode=2 # 1: (a+5)/10, #2: MinMaxScaler, 3: noting 


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






Test_network = SparseEstimatorNetwork(img_shape=input_shape, encoded_dim=encoded_dim,
                                      Number_of_pilot=Number_of_pilot,regularizer_coef=regularizer_coef,
                                      on_cloud=on_cloud,test_mode =1 , log_path=log_path, normalize_mode=normalize_mode,
                                      Noise_var_L=Noise_var_L, Noise_var_H=Noise_var_H, data_type=data_type, 
                                      Enable_conv=Enable_conv,Fixed_pilot=Fixed_pilot,Enable_auto=Enable_auto,Drou_out_sel=Drou_out_sel)


import scipy.io
import io
import base64
@app.route("/estimate_channel", methods = ['POST'])
def estimate_channel():
    global normalize_mode
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
    elif normalize_mode==5:
      image= (image+5)
    elif normalize_mode==4:
    	image= (image)/5



    y = Test_network.test(image, var)
    
    if normalize_mode==1:
      y= y*10-5
    elif normalize_mode==5:
      y= y-5
    elif normalize_mode==4:
    	y=y*5

    output = io.BytesIO()
    o ={}
    o['out']=y[0]
    scipy.io.savemat(output, o)
    return base64.b64encode(output.getvalue())


@app.route("/estimate_channel_vjason", methods = ['POST'])
def estimate_channel_vjason():
  global Test_network
  global normalize_mode
  data = json.loads(request.data.decode('utf-8'))
  #print(data)

  image=np.array(data['image'])
  Noise_var = data['Noise_var']
  #Noise_var=pow(10,(-50/10))
  #print(Noise_var)
  #print("-----------")
  #print(image)
  if normalize_mode==1:
    image= (image+5)/10.0
  elif normalize_mode==5:
    image= (image+5)
  elif normalize_mode==4:
    image= (image)/5

  image=image.reshape(1,image.shape[0],image.shape[1])
  y, y_intrpolated,y_ConvOut = Test_network.test(image, Noise_var)
  if normalize_mode==1:
    y= y*10-5
  elif normalize_mode==5:
    y= y-5
  elif normalize_mode==4:
  	y=y*5

  result = json.dumps(y[0].tolist())
  return result

if __name__ == "__main__":
	app.run()	