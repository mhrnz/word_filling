
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import pickle
import import_ipynb
import biLstm_word_filling
import numpy.random as random


# In[3]:


'''
description: deletes words of the test corpus with probability 0.03, 
             deleted words are treated as blanks with random embeddings( different random embeddings for different blanks)
input: int_words: list of word indices in glove_vectors , glove_vectors size, window_size , 
       int_to_vocab : a dictionary from indices to words
'''
def delete_some_words(int_words,glove_vectors_size,window_size,int_to_vocab):
    blank_indices=[]
    '''a dictionary from blank index in int_words to its real index in glove_vectors'''
    real_words={}
    for i in range(window_size//2,len(int_words)-(window_size//2)+1):
        if random.uniform()<0.03:
            '''
            int_words[i] is equal to an index in glove_vectors, we should keep the real index of the deleted word 
            (which will refer to a new index(at the end of the glove_vectors) after deletion) in order to retrieve it
            '''
            int_to_vocab[glove_vectors_size]=int_to_vocab[int_words[i]]
            real_words[i]=int_words[i]
            int_words[i]=glove_vectors_size
            glove_vectors_size+=1
    return int_words,glove_vectors_size,real_words,int_to_vocab


# In[4]:


def test(int_words,glove_vectors,int_to_vocab,modelpath,window_size=101,num_layers=2,hidden_size=200,keep_prob=1,batch_size=1):
        
    vocab_size=glove_vectors.shape[0]
    embedding_dim=glove_vectors.shape[1]
    '''building bidirectional lstm model with appropriate parameters'''
    model=biLstm_word_filling.biLstm_word_filling(window_size=window_size,num_layers=num_layers,embedding_dim=embedding_dim,hidden_size=hidden_size,keep_prob=keep_prob,vocab_size=vocab_size)
    saver=tf.train.Saver()
    file=open('result_1.txt',"w")
    with tf.Session() as sess:
        '''restoring from saved model'''
        saver.restore(sess,modelpath)
        int_words,glove_vectors_size,real_words,int_to_vocab=delete_some_words(int_words=int_words,glove_vectors_size=vocab_size,window_size=window_size,int_to_vocab=int_to_vocab)
        '''num_blanks is equal to number of deleted words'''
        num_blanks=glove_vectors_size-vocab_size

        x=np.zeros((1,window_size),dtype=np.int32)
        '''random_embeddings for deleted words'''
        randomblanks=np.random.normal(size=(num_blanks,embedding_dim))
        i=0
        min_loss=12000
        correct=0
        total=0
        mean_loss=0
        def coef(x):
            '''assign a coeficient p to frequent words
            0:the -> p=0.7 , 1:, -> p=0.5 , 2:. -> p=0.5 , 3:of -> p=0.7 , 4:to -> p=0.7 , 5:and -> p=0.7
            '''
            return {0:0.7 , 1:0.5 , 2:0.5 , 3:0.7 , 4:0.7 , 5:0.7}[x]
        
        for blank_index,real_int in real_words.items():
            '''batch_size=1 , for each blank'''
            x[0]=np.copy(int_words[blank_index-(window_size//2):blank_index-(window_size//2)+window_size])
            y=np.array([real_int])
            frequent_words_coefs=np.ones(y.shape)
            if y[i] in range(6):
                frequent_words_coefs[i]=coef(y[i])
            feed_dict={model.inputs:x,model.output:y,model.random_blanks:randomblanks,model.frequent_words_coefs:frequent_words_coefs}
            loss,logits=sess.run([model.loss,model.logits],feed_dict=feed_dict)
            mean_loss+=loss
            min_loss=min(loss,min_loss)
            i+=1
            predicted_index=tf.argmax(logits,axis=1).eval()
            input_words=[int_to_vocab[i] for i in x[0]]
            input_words[window_size//2]='<blank>'
            file.write('i:'+str(i)+' loss:'+str(loss))
            file.write('\n')
            print('i:',i,' loss:',loss)
            file.write('input_words:'+str(input_words))
            file.write('\n')
            print('input_words:',input_words)
            file.write('label:'+str(int_to_vocab[real_int]))
            file.write('\n')
            print('label:',int_to_vocab[real_int])
            file.write('predicted_word:'+str(int_to_vocab[predicted_index[0]]))
            file.write('\n')
            print('predicted_word:',str(int_to_vocab[predicted_index[0]]))
            total+=1
            if real_int==predicted_index[0]:
                correct+=1
        mean_loss=mean_loss/i
        file.write('correct_factor:'+str(correct/total))
        print('correct_factor:',str(correct/total))
        file.write('mean_loss:'+str(mean_loss))
        print('mean_loss:',mean_loss)
        file.write('min_loss:'+str(min_loss))
        print('min_loss:',min_loss)
        file.close()
                    
                    
                    


# In[5]:


if __name__=="__main__":
    with open('int_words_test.pickle','rb') as handle:
        int_words=pickle.load(handle)
    with open('glove_vectors.pickle','rb') as handle:
        glove_vectors=pickle.load(handle)
        print('glove_vectors shape:',glove_vectors.shape)
    with open('int_to_vocab_test.pickle','rb') as handle:
        int_to_vocab=pickle.load(handle)
    test(int_words,glove_vectors,int_to_vocab,modelpath='./trained_word_filling.ckpt')

