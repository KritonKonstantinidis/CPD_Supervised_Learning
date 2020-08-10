import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math

def simple_batcher(x, y, bs):
    for begin, end in zip(range(0, len(x), bs)[:-1], range(0, len(x), bs)[1:]):
        yield (x[begin:end], y[begin:end])

class Tree_Machine:
    def __init__(self, rank=5, s_features=[2, 3, 4, 5], init_std=0.7, exp_reg=1.1, reg=0.01, seed=42):
        self.rank =  rank
        self.s_features = s_features
        self.n_features = len(s_features)
        self.init_std = init_std
        self.graph = tf.Graph()
        if seed:
            self.graph.seed = seed
        self.exp_reg = exp_reg
        self.reg = reg
        self.init_vals=None # matrix values
        self.init_cores=None        
        
    def init_from_cores(self, core_list):
        assert(len(core_list) == len(self.s_features))
        self.init_vals = core_list
        
    def init_from_cores_tensors(self,tensors_list):
        self.init_cores=tensors_list
        
    def build_graph(self):
        with self.graph.as_default():
            
            # placeholders
            self.X = tf.placeholder(tf.int64, [None, self.n_features], name='X')
            self.Y = tf.placeholder(tf.float32, (None), name='Y')
            
            ### Setting the values for factor matrices 

            # list of factor matrices 
            self.G = [None]*self.n_features

            # list of factor matrices used for penalty
            self.G_exp = [None]*self.n_features
            
            for i in range(self.n_features):

                shape = [self.s_features[i] + 1, self.rank]

                if self.init_vals is None:
                    content = tf.random_normal(shape, stddev=self.init_std)
                else:
                    assert(self.init_vals[i].shape==tuple(shape))
                    content = self.init_vals[i] + tf.random_normal(shape, stddev=self.init_std)
               
                self.G[i] = tf.Variable(content, trainable=True, name='G_{}'.format(i))
                
                # regularization setup
                exp_weights = tf.constant([1] + [self.exp_reg] * self.s_features[i], shape=(self.s_features[i] + 1, 1))
                self.G_exp[i] = self.G[i] * exp_weights
            
            ### Setting the values for tensor cores 
            
            levels=math.ceil(np.log2(self.n_features)) # num levels, in this case 5 levels, excluding factor matrices
            num_tensors_last_level=self.n_features-2**(levels-1) # 10 in this case
            
            self.G_tensors=[]
            for i in range(levels):
                G_tensors_local=[]
                for j in range(2**i):
                    if (i==levels-1) and (j==num_tensors_last_level):  # if in the last level and last tensor, break
                        break
                    if(i==0):
                        G_tensors_local.append(tf.Variable(self.init_cores[i][j]+tf.random_normal(shape=(self.rank,self.rank)
                        ,stddev=self.init_std), trainable=True, name='G_tensor{}{}'.format(i,j)))
                    else:
                        G_tensors_local.append(tf.Variable(self.init_cores[i][j]+tf.random_normal(shape=(self.rank,self.rank,self.rank)
                        , stddev=self.init_std), trainable=True, name='G_tensor{}{}'.format(i,j)))
                        
                self.G_tensors.append(G_tensors_local)
            
            
            ############## Forward Pass ###########
            
            # Contraction at lower level
            
            previous_vectors=[]
            new_vectors=[]
            
            cur_col = self.X[:, 0]
            tower = tf.gather(self.G[0], cur_col)
            previous_vectors.append(tf.add(self.G[0][0], tower))
            
            for i in range(1, self.n_features):
                cur_col = self.X[:, i] # batch corresponding to feature i
                cur_tower = tf.gather(self.G[i], cur_col) 
                if i<2*len(self.G_tensors[levels-1]):
                    previous_vectors.append(tf.add(self.G[i][0], cur_tower))
                else:
                    new_vectors.append(tf.add(self.G[i][0], cur_tower))
            
            # Contraction at higher levels 
            for i in range(levels-1,-1,-1):
                for j in range(len(self.G_tensors[i])-1,-1,-1):
                    if(i==0):
                        first_con=tf.matmul(previous_vectors[2*j],self.G_tensors[i][j])    # (Ntxm)x(mxm), output is Ntxm
                        first_con=tf.expand_dims(first_con,axis=1) # Ntx1xm
                        previous_vectors[2*j+1]=tf.expand_dims(previous_vectors[2*j+1],axis=2) # Ntxm becomes Ntxmx1 

                        second_con=tf.matmul(first_con,previous_vectors[2*j+1],) # (Ntx1xm) x (Ntxmx1), output is Ntx1x1
                        second_con=tf.squeeze(second_con) # Ntx1
                        
                    else:
                        self.G_tensors[i][j]=tf.reshape(self.G_tensors[i][j],shape=(self.rank,self.rank**2)) # mxm^2
                        first_con=tf.matmul(previous_vectors[2*j],self.G_tensors[i][j]) # (Ntxm)x(mxm^2), output is Ntxm^2
                        first_con=tf.reshape(first_con,shape=(-1,self.rank,self.rank)) # Ntxmxm 
                        previous_vectors[2*j+1]=tf.expand_dims(previous_vectors[2*j+1],axis=1) # Ntx1xm
                    
                        second_con=tf.matmul(previous_vectors[2*j+1],first_con) # (Ntx1xm) x (Ntxmxm), output is Ntx1xm
                        second_con=tf.squeeze(second_con,axis=1) # Ntxm 

                    new_vectors.insert(0,second_con) 

                    
                previous_vectors=new_vectors
                new_vectors=[]
            self.outputs=previous_vectors[0] # scalar
                    
            
            ######### Regularization ##########
#            
            previous_matrices=[]
            new_matrices=[]
            
            for i in range(0, self.n_features):
                
                if i<2*len(self.G_tensors[levels-1]):
                    previous_matrices.append(tf.matmul(tf.transpose(self.G_exp[i]),self.G_exp[i])) 
                else:
                    new_matrices.append(tf.matmul(tf.transpose(self.G_exp[i]),self.G_exp[i]))
                    
            for i in range(levels-1,-1,-1):
                for j in range(len(self.G_tensors[i])-1,-1,-1):
                    
                    if(i==0):
                        
                        first_con=tf.matmul(previous_matrices[2*j],self.G_tensors[i][j]) # (mxm) x (mxm), output is mxm 
                        second_con=tf.matmul(first_con,previous_matrices[2*j+1]) # (mxm) x (mxm), output is mxm 
                        third_con=tf.matmul(second_con,self.G_tensors[i][j])
                        
                    else:
                        
                        tensor_reshaped=tf.reshape(self.G_tensors[i][j],shape=(self.rank,self.rank**2)) # mxm^2
                        
                        first_con=tf.matmul(previous_matrices[2*j],tensor_reshaped) # (mxm)x(mxm^2), output is mxm^2
                        
                        first_con=tf.reshape(first_con,shape=(self.rank**2,self.rank)) # m^2xm 
                                 
                        second_con=tf.matmul(previous_matrices[2*j+1],tf.transpose(first_con)) # (mxm) x (mxm^2), output is mxm^2
                        
                        second_con=tf.reshape(second_con,shape=(self.rank**2,self.rank)) 
                        
                        third_con =tf.matmul(tf.transpose(second_con),tf.reshape(self.G_tensors[i][j],shape=(self.rank**2,self.rank))) # (mxm^2)x  (m^2xm), output is mxm

                    new_matrices.insert(0,third_con) 
                    
                previous_matrices=new_matrices
                new_matrices=[]

            self.penalty=tf.trace(previous_matrices[0])

            self.loss = tf.reduce_mean((self.outputs - self.Y)**2)
            self.penalized_loss = self.loss + self.reg * self.penalty

            # others
            self.trainer = tf.train.AdamOptimizer(0.00005).minimize(self.penalized_loss) 
            # for increased stability, higher learning rates were found to lead to occasionally unstable results

            self.init_all_vars = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

    def initialize_session(self):
        config = tf.ConfigProto()
        # for reduce memory allocation
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.init_all_vars)

    def destroy(self):
        self.session.close()
        self.graph = None