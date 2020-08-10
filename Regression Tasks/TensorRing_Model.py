import tensorflow as tf
from tensorflow import keras

class TensorRing_Based(keras.layers.Layer):

    def __init__(self, units=1, activation=None, rank=10, local_dim=2,initializer=keras.initializers.glorot_normal(seed=None),regularizer=keras.regularizers.l2(0.0), **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.rank = rank
        self.local_dim = local_dim
        self.initializer=initializer
        self.kernel_regularizer = regularizer

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[self.local_dim, self.rank, self.rank, batch_input_shape[-2], self.units],
                                      initializer=self.initializer,regularizer=self.kernel_regularizer) 
        super().build(batch_input_shape)

    def call(self, X):
            
        feat_tensor=X # NtxNxd
        
        output_list=[]
        
        for unit in range(0,self.units):    

            feat_tensor_reshaped=tf.transpose(feat_tensor,perm=[1,0,2]) # NxNtxd
            weights=self.kernel[:,:,:,:,unit] # initial shape of weights is dxmxmxN
            weights=tf.transpose(self.kernel[:,:,:,:,unit],perm=[3,0,1,2]) # Nxdxmxm
            weights=tf.reshape(weights,shape=(-1,self.local_dim,self.rank**2)) # Nxdxm^2
            
            test=tf.matmul(feat_tensor_reshaped,weights) # NxNtxm^2
            
            test=tf.reshape(test,shape=(tf.shape(self.kernel)[3],-1,self.rank,self.rank))  #NxNtxmxm
            
           # N mxm matrix multiplications
            output=test[0,:,:,:]
            for i in range(1,8):
                output=tf.matmul(output,test[i,:,:,:])
            
            # output now is Ntxmxm
            
            output=tf.linalg.trace(output) # Ntx1
            output_list.append(output)
        
        to_return=tf.stack(output_list, axis=1)
        return self.activation(to_return)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
        
class OurModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer1 = AllOrder()
        
    def call(self, inputs):
        return self.layer1(inputs)
