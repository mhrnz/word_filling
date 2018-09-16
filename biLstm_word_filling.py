
# coding: utf-8

# In[2]:

# do not use extra packages
import tensorflow as tf
import collections
import tensorflow.contrib.rnn as rnn
import torchtext.vocab as vocab
import numpy as np
# import nltk
import pickle
# import import_ipynb
import data_preprocessing



# In[3]:


# bidirectional lstm model for word filling
# assumes center word as blank, predicts it and calculates loss for that
class biLstm_word_filling():
    def __init__(self,window_size,num_layers,vocab_size,embedding_dim,hidden_size,keep_prob):
    """
    Description:
    ...

    input:
    ...

    output:
    ...
    """
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.glove_weights = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),trainable=False, name="glove_weights")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim],name="embedding_placeholder")
        self.embedding_init = self.glove_weights.assign(self.embedding_placeholder)

        self.random_blanks=tf.placeholder(tf.float32,shape=[None,None],name='random_blanks')
        # adding random blank embeddings to the end of out embedding_init matrix
        self.embedding_with_randoms=tf.concat([self.glove_weights,self.random_blanks],axis=0)

        self.inputs=tf.placeholder(tf.int32,shape=[None,None],name='inputs')
        self.output=tf.placeholder(tf.int32,shape=[None],name='output') # center word of the window
        self.batch_size=tf.shape(self.inputs)[0]

#         self.init_state=tf.placeholder(tf.float32,[self.num_layers,2,self.batch_size,self.hidden_size])
#         state_per_layer_list = tf.unstack(self.init_state, axis=0)
#         rnn_tuple_state = tuple([rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_layers)])

        with tf.name_scope("embedding_lookup"):
            # getting embedding vectors of input indices from embedding_init
            self.embeds=tf.nn.embedding_lookup(self.embedding_with_randoms,self.inputs)
#             print(type(self.embeds))
#             self.embeds[:,int(window_size//2),:].assign(tf.reduce_mean(self.embeds[:,int(window_size//2-5):int(window_size//2),:],axis=1)+tf.reduce_mean(self.embeds[:,int(window_size//2+1):int(window_size//2+6),:],axis=1))

        with tf.name_scope("bi_lstm"):
            def make_cell():
                cell=rnn.BasicLSTMCell(self.hidden_size)
                cell=rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
                return cell

            # forward and backward cell
            fw_cell=rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
            bw_cell=rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
            lstm_outputs,self.last_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embeds,dtype=tf.float32)

            fw_output=lstm_outputs[0][:,int(window_size//2),:]
            bw_output=lstm_outputs[1][:,int(window_size//2),:]
            # predicted_outputs before softmax, from hiddens
            predicted_output=tf.concat([fw_output,bw_output],axis=1)

            # logits : densed predicted_output to vocab_size so we can pass it to softmax cross entropy loss
            self.logits = tf.layers.dense(predicted_output, vocab_size)
            self.one_hot_outputs=tf.one_hot(indices=self.output,depth=vocab_size,axis=-1)
            # can't you use index instead of one hot vector ?
            with tf.name_scope("loss"):
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits,
                labels=self.one_hot_outputs,
                dim=1)
            self.cost=tf.reduce_mean(self.loss)
            self.optimizer=tf.train.AdamOptimizer(learning_rate=0.02).minimize(self.cost)
