import tensorflow as tf
import keras 

class TensorTree_Based(keras.layers.Layer):

    def __init__(self, units=1, activation=None, rank=10, local_dim=2,initializer=keras.initializers.glorot_normal(seed=None),regularizer=keras.regularizers.l2(0.0), **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.rank = rank
        self.local_dim = local_dim
        self.initializer=initializer
        self.regularizer = regularizer

    def build(self, batch_input_shape): 
        
        self.core11 = self.add_weight(name="core11", shape=[self.rank, self.rank, self.rank],
                                     initializer=self.initializer,regularizer=self.regularizer,trainable=True)
        self.core12 = self.add_weight(name="cores12", shape=[self.rank, self.rank, self.rank],
                                     initializer=self.initializer,regularizer=self.regularizer,trainable=True)
        self.core13 = self.add_weight(name="cores13", shape=[self.rank, self.rank, self.rank],
                                     initializer=self.initializer,regularizer=self.regularizer,trainable=True)
        self.core14 = self.add_weight(name="cores14", shape=[self.rank, self.rank, self.rank],
                                     initializer=self.initializer,regularizer=self.regularizer,trainable=True)
        self.core21 = self.add_weight(name="cores21", shape=[self.rank, self.rank, self.rank],
                                       initializer=self.initializer,regularizer=self.regularizer,trainable=True)
        self.core22 = self.add_weight(name="cores22", shape=[self.rank, self.rank, self.rank],
                                       initializer=self.initializer,regularizer=self.regularizer,trainable=True)
        
        self.factor_matrices = self.add_weight(name="matrices", shape=[self.local_dim, self.rank, 8, self.units],
                                               initializer=self.initializer,regularizer=self.regularizer,trainable=True) 
        
        self.matrix_top=self.add_weight(name="matrix_top", shape=[self.rank, self.rank],
                                       initializer=self.initializer,regularizer=self.regularizer,trainable=True)
        super().build(batch_input_shape)

    def call(self, X):
            
        feat_tensor=X # NtxNxd
        
        output_list=[]
        feat_tensor_reshaped=tf.transpose(feat_tensor,perm=[1,0,2]) # NxNtxd

### Contraction at the first level ###
        
        for unit in range(0,self.units):    

            weights=tf.transpose(self.factor_matrices[:,:,:,unit],perm=[2,0,1]) # Nxdxm
            contraction1=tf.matmul(feat_tensor_reshaped,weights) # NxNtxm
        
### Contraction at the second level ###
            
            self.core11=tf.reshape(self.core11,shape=(self.rank,self.rank**2)) # mxm^2
            self.core12=tf.reshape(self.core12,shape=(self.rank,self.rank**2)) # mxm^2
            self.core13=tf.reshape(self.core13,shape=(self.rank,self.rank**2)) # mxm^2
            self.core14=tf.reshape(self.core14,shape=(self.rank,self.rank**2)) # mxm^2
            self.core21=tf.reshape(self.core21,shape=(self.rank,self.rank**2)) # mxm^2
            self.core22=tf.reshape(self.core22,shape=(self.rank,self.rank**2)) # mxm^2

            # core 1 
            con21=tf.matmul(contraction1[0,:,:],self.core11) # (Ntxm)x(mxm^2), output is Ntxm^2
            con21=tf.reshape(con21,shape=(-1,self.rank,self.rank)) # Ntxmxm 
            to_contract_1=tf.expand_dims(contraction1[1,:,:],axis=1) # Ntxm becomes Ntx1xm
            
            con21=tf.matmul(to_contract_1,con21) # (Ntx1xm)x(Ntxmxm), output is Ntx1xm
            con21=tf.squeeze(con21,axis=1) # Ntxm 
            
            # core 2
            con22=tf.matmul(contraction1[2,:,:],self.core12) # (Ntxm)x(mxm^2), output is Ntxm^2
            con22=tf.reshape(con22,shape=(-1,self.rank,self.rank)) # Ntxmxm 
            to_contract_2=tf.expand_dims(contraction1[3,:,:],axis=1) # Ntxm becomes Ntx1xm
            con22=tf.matmul(to_contract_2,con22) # (Ntx1xm)x(Ntxmxm), output is Ntx1xm
            con22=tf.squeeze(con22,axis=1) # Ntxm 
            
             # core 3
            con23=tf.matmul(contraction1[4,:,:],self.core13) # (Ntxm)x(mxm^2), output is Ntxm^2
            con23=tf.reshape(con23,shape=(-1,self.rank,self.rank)) # Ntxmxm 
            to_contract_3=tf.expand_dims(contraction1[5,:,:],axis=1) # Ntxm becomes Ntx1xm
            con23=tf.matmul(to_contract_3,con23) # (Ntx1xm)x(Ntxmxm), output is Ntx1xm
            con23=tf.squeeze(con23,axis=1) # Ntxm 
            
             # core 4
            con24=tf.matmul(contraction1[6,:,:],self.core14) # (Ntxm)x(mxm^2), output is Ntxm^2
            con24=tf.reshape(con24,shape=(-1,self.rank,self.rank)) # Ntxmxm 
            to_contract_4=tf.expand_dims(contraction1[7,:,:],axis=1) # Ntxm becomes Ntx1xm
            
            con24=tf.matmul(to_contract_4,con24) # (Ntx1xm)x(Ntxmxm), output is Ntx1xm
            con24=tf.squeeze(con24,axis=1) # Ntxm 
            
### Contraction at the third level 
            
            # Core 1
            con31=tf.matmul(con21,self.core21) # (Ntxm)x(mxm^2), output is Ntxm^2
            con31=tf.reshape(con31,shape=(-1,self.rank,self.rank)) # Ntxmxm 
            to_contract_5=tf.expand_dims(con22,axis=1) # Ntxm becomes Ntx1xm
            con31=tf.matmul(to_contract_5,con31) # (Ntx1xm)x(Ntxmxm), output is Ntx1xm
            con31=tf.squeeze(con31,axis=1) # Ntxm 
            
            # Core 2
            con32=tf.matmul(con23,self.core22) # (Ntxm)x(mxm^2), output is Ntxm^2
            con32=tf.reshape(con32,shape=(-1,self.rank,self.rank)) # Ntxmxm 
            to_contract_6=tf.expand_dims(con24,axis=1) # Ntxm becomes Ntx1xm
            con32=tf.matmul(to_contract_6,con32) # (Ntx1xm)x(Ntxmxm), output is Ntx1xm
            con32=tf.squeeze(con32,axis=1) # Ntxm 
    
### Contraction at the 4th and final level 
            con4=tf.matmul(con31,self.matrix_top)    # (Ntxm)x(mxm), output is Ntxm
            con4=tf.expand_dims(con4,axis=1) # Ntxm becomes Ntx1xm
            con32=tf.expand_dims(con32,axis=2) # Ntxm becomes Ntxmx1 

            output= tf.matmul(con4,con32) # (Ntx1xm) x (Ntxmx1), output is Ntx1x1
            output=tf.squeeze(output) # Ntx1

            output_list.append(output)
            
        to_return=tf.stack(output_list, axis=1)
        return self.activation(to_return)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
        
