import tensorflow as tf
import keras

class Tucker_Based(keras.layers.Layer):

    def __init__(self, units=1, activation=None, Tucker_rank=10, local_dim=2,initializer=keras.initializers.glorot_normal(seed=None),regularizer=keras.regularizers.l2(0.0), **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.Tucker_rank = Tucker_rank
        self.local_dim = local_dim
        self.initializer=initializer
        self.kernel_regularizer = regularizer

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[self.local_dim, self.Tucker_rank, batch_input_shape[-2], self.units],
                                      initializer=self.initializer,regularizer=self.kernel_regularizer)
        
        self.core = self.add_weight(name="core",shape=[self.Tucker_rank,self.Tucker_rank,self.Tucker_rank,self.Tucker_rank,
                                                        self.Tucker_rank,self.Tucker_rank,self.Tucker_rank,self.Tucker_rank], 
                                    initializer=self.initializer,regularizer=self.kernel_regularizer)
        super().build(batch_input_shape)

    def call(self, X):
            
        feat_tensor=X
        
        output_list=[]
        
        for unit in range(0,self.units):                 
                    
            feat_tensor_reshaped=tf.transpose(feat_tensor,perm=[1,0,2]) # NxNtxd
            weights=tf.transpose(self.kernel[:,:,:,unit],perm=[2,0,1]) # Nxdxm
            test=tf.matmul(feat_tensor_reshaped,weights) # NxNtxm
            
            # Contract factor matrices with core tensor
            output=tf.reshape(self.core,shape=[self.Tucker_rank,-1]) #mxm^7
            output=tf.matmul(test[0,:,:],output) #  (Ntxm) x(mxmx^7), output is Ntxm^7

            output=tf.reshape(output,shape=[-1,self.Tucker_rank,self.Tucker_rank**6]) # Ntxmxm^6
            output=tf.matmul(tf.expand_dims(test[1,:,:],axis=1),output) # (Ntx1xm) x (Ntxmxm^6), output is Ntx1xm^6
            output=tf.squeeze(output) # Ntxm^6
            
            output=tf.reshape(output,shape=[-1,self.Tucker_rank,self.Tucker_rank**5]) # Ntxmxm^6
            output=tf.matmul(tf.expand_dims(test[2,:,:],axis=1),output) #  (Ntx1xm) x (Ntxmxm^5), output is Ntx1xm^5
            output=tf.squeeze(output) # Ntxm^5
            
            output=tf.reshape(output,shape=[-1,self.Tucker_rank,self.Tucker_rank**4]) # Ntxmxm^6
            output=tf.matmul(tf.expand_dims(test[3,:,:],axis=1),output) # (Ntx1xm) x (Ntxmxm^4), output is Ntx1xm^4
            output=tf.squeeze(output) # Ntxm^4
            
            output=tf.reshape(output,shape=[-1,self.Tucker_rank,self.Tucker_rank**3]) # Ntxmxm^6
            output=tf.matmul(tf.expand_dims(test[4,:,:],axis=1),output) # (Ntx1xm) x (Ntxmxm^3), output is Ntx1xm^3
            output=tf.squeeze(output) # Ntxm^3
            
            output=tf.reshape(output,shape=[-1,self.Tucker_rank,self.Tucker_rank**2]) # Ntxmxm^6
            output=tf.matmul(tf.expand_dims(test[5,:,:],axis=1),output) # (Ntx1xm) x (Ntxmxm^2), output is Ntx1xm^2
            output=tf.squeeze(output) # Ntxm^2
            
            output=tf.reshape(output,shape=[-1,self.Tucker_rank,self.Tucker_rank]) # Ntxmxm^6
            output=tf.matmul(tf.expand_dims(test[6,:,:],axis=1),output) # (Ntx1xm) x (Ntxmxm), output is Ntx1xm
            output=tf.squeeze(output) # Ntxm
            
            output=tf.matmul(tf.expand_dims(test[7,:,:],axis=1),tf.expand_dims(output,axis=2)) # (Ntx1xm) x (Ntxmx1), output is Ntx1x1
            output=tf.squeeze(output) # Ntx1

        output_list.append(output)
        
        to_return=tf.stack(output_list, axis=1)
        return self.activation(to_return)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
        
