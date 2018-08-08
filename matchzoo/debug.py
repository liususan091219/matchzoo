from keras import backend as K
from keras.callbacks import Callback
import numpy as np
            
def calc_stats(W):
    return np.linalg.norm(W, 2), np.mean(W), np.std(W)

class MyDebugWeights(Callback):
    
    def __init__(self, model):
        super(MyDebugWeights, self).__init__()
        self.weights = {} 
        self.tf_session = K.get_session()
	self.model = model
            
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            name = layer.name
            for i, w in enumerate(layer.weights):
                w_value = w.eval(session=self.tf_session)
                w_norm, w_mean, w_std = calc_stats(np.reshape(w_value, -1))
                self.weights.setdefault((i, w), [])
		self.weights[(i, w)].append((epoch, "{:s}/W_{:d}".format(name, i), 
                                     w_norm, w_mean, w_std))
    
    def on_train_end(self, logs=None):
        for e, k, n, m, s in self.weights:
            print("{:3d} {:20s} {:7.3f} {:7.3f} {:7.3f}".format(e, k, n, m,s ))
