from flask import Flask
from flask import request
import json
import numpy as np
from ChannelEstimatorNetwork import SparseEstimatorNetwork
app = Flask(__name__)
 

network = SparseEstimatorNetwork(img_shape=(72, 72), encoded_dim=600)
network.autoencoder.load_weights('model.h5')

@app.route("/estimate_channel", methods = ['POST'])
def estimate_channel():
    channel = np.array(json.loads(request.data))
    output = network.autoencoder.predict(channel.reshape((1,)+channel.shape))
    result = json.dumps(output[0].tolist())
    return result

if __name__ == "__main__":
    app.run()