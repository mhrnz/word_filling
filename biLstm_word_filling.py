
# coding: utf-8

# In[7]:


import tensorflow as tf
import tensorflow.contrib.rnn as rnn


# In[8]:


'''
bidirectional lstm model for word filling
assumes center word as blank, predicts it and calculates loss for that
'''
class biLstm_word_filling():
    def __init__(self,window_size,num_layers,vocab_size,embedding_dim,hidden_size,keep_prob):
        
        with tf.device('/gpu:0'):
        
            self.num_layers=num_layers
            self.hidden_size=hidden_size
            self.glove_weights = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),trainable=False, name="glove_weights")
            self.embedding_placeholder = tf.placeholder(tf.float32, [None, embedding_dim],name="embedding_placeholder")
            self.embedding_init = self.glove_weights.assign(self.embedding_placeholder)

            self.random_blanks=tf.placeholder(tf.float32,shape=[None,None],name='random_blanks')
            '''adding random blank embeddings to the end of our embedding_init matrix'''
            self.embedding_with_randoms=tf.concat([self.glove_weights,self.random_blanks],axis=0)

            self.inputs=tf.placeholder(tf.int32,shape=[None,None],name='inputs')
            self.output=tf.placeholder(tf.int32,shape=[None],name='output') # center word of the window
            '''coeficients for labels. less han 1 for 5 most frequent words, 1 for others'''
            self.frequent_words_coefs=tf.placeholder(tf.float32,shape=[None],name='frequent_words_coefs')
            self.batch_size=tf.shape(self.inputs)[0]
            
            with tf.name_scope("embedding_lookup"):
                '''getting embedding vectors of input indices from embedding_init'''
                self.embeds=tf.nn.embedding_lookup(self.embedding_with_randoms,self.inputs)
                
            with tf.name_scope("bi_lstm"):
                def make_cell():
                    cell=rnn.BasicLSTMCell(self.hidden_size)
                    cell=rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
                    return cell

                # forward and backward cell
#                 fw_cell=rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
                fw_cell=make_cell() # one layer lstm
                bw_cell=make_cell() # one layer lstm
#                 bw_cell=rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
                lstm_outputs,self.last_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embeds,dtype=tf.float32)

                fw_output=lstm_outputs[0][:,int(window_size//2),:]
                bw_output=lstm_outputs[1][:,int(window_size//2),:]
                '''predicted_outputs before softmax, from hiddens'''
                predicted_output=tf.concat([fw_output,bw_output],axis=1)

                '''logits : densed predicted_output to vocab_size so we can pass it to softmax cross entropy loss'''
                self.logits = tf.layers.dense(predicted_output, vocab_size,name='logits')
                self.one_hot_outputs=tf.one_hot(indices=self.output,depth=vocab_size,axis=-1)

                with tf.name_scope("loss"):
                    self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits,
                    labels=self.one_hot_outputs,
                    dim=1,
                    name='loss')
                    
                self.loss=tf.multiply(self.loss,self.frequent_words_coefs)
                self.cost=tf.reduce_mean(self.loss,name='cost')
                
#                 trainable_params=tf.trainable_variables()
#                 clip_gradients=[tf.clip_by_norm(grads,self.max_gradient_norm) for grads in tf.gradients(self.loss,trainable_params)] 
                
                self.new_lr=tf.placeholder(tf.float32,shape=[])
                self.learning_rate=tf.Variable(0.0,trainable=False,name='learning_rate')
                self.update_lr=self.learning_rate.assign(self.new_lr)
                self.optimizer=tf.train.AdamOptimizer(self.learning_rate,name='optimizer').minimize(self.cost)
#                 self.updates = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))

