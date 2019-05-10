import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, GRU, Masking, Lambda, Bidirectional, Dropout, Reshape
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import keras.backend as K
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from EpisodicMemoryModule import EpisodicMemoryModule

class DynamicNetworkMemory():
    def __init__(self, max_context_len, max_question_len, max_sent_num, output_dim, \
                 word_vector_size=50, num_memory_passes = 4, regularization_val = 1e-4,\
                 hidden_units = 100, batch_size = 50, dropout = 0.1, learning_rate = 0.001,\
                summary = True):
        self.word_vector_size   = word_vector_size
        self.max_context_len    = max_context_len
        self.max_question_len   = max_question_len
        self.max_sent_num       = max_sent_num
        self.hidden_units       = hidden_units
        self.input_h_units      = int(self.hidden_units/2)
        self.dropout            = dropout
        self.regularization_val = regularization_val
        self.num_memory_passes  = num_memory_passes
        self.batch_size         = batch_size
        self.output_dim         = output_dim
        self.learning_rate      = learning_rate
        self.summary            = summary
      
    def get_facts(self, facts_output, sent_end_idxs):    
        def extract_facts(input_tuple):
            timesteps, padded_indices = input_tuple 
            num_facts = tf.count_nonzero(input_tensor=padded_indices, dtype=tf.int32, keep_dims=True) 
            indices = tf.slice(input_=padded_indices, begin=tf.zeros([1,], tf.int32), size=num_facts)

            facts_tensor = tf.nn.embedding_lookup(timesteps, ids=indices)

            pad = tf.zeros(shape=[self.max_sent_num-tf.shape(facts_tensor)[0], \
                                  self.hidden_units], dtype=tf.float32)
            padded_facts_tensor = K.concatenate([facts_tensor, pad], axis=0)

            return padded_facts_tensor
    
        input_tuple = (facts_output, sent_end_idxs)
        padded_fact_tensors = tf.map_fn(fn=extract_facts, elems=input_tuple, dtype=tf.float32)  

        return padded_fact_tensors
        
    def get_model(self):       
        # def context_module(self):
        context_input = Input(shape=(self.max_context_len, self.word_vector_size), name="ContextInput")
        context_mask = Masking(mask_value=0.0, name="ContextMask")(context_input)
        facts_output = Bidirectional(GRU(units = self.input_h_units, return_sequences=True, \
                                         dropout = self.dropout, merge_mode="concat", \
                                         kernel_regularizer=regularizers.l2(self.regularization_val),\
                                         recurrent_regularizer=regularizers.l2(self.regularization_val)),\
                                     name="ContextBiGRU")(context_mask)
        
        sent_end_idx_input = Input(shape=(self.max_sent_num, ), dtype=tf.int32, name="IndicesInput") 
        facts_tensors = Lambda(self.get_facts, arguments={"sent_end_idxs": sent_end_idx_input},
                                                   name="FactTensorLambda")(facts_output)
        facts_tensors = Reshape((self.max_sent_num, 2*self.input_h_units),\
                                name="Fact_Reshape")(facts_tensors)
        
    
        #def question_module(self):
        
        question_input = Input(shape=(self.max_question_len, self.word_vector_size), name="QuestionInput")

        question_mask = Masking(mask_value=0.0, name="QuestionMask")(question_input)

        question_output = GRU(units = self.hidden_units, dropout=self.dropout,
                              kernel_regularizer=regularizers.l2(self.regularization_val),
                              recurrent_regularizer=regularizers.l2(self.regularization_val),
                              name="QuestionGRU")(question_mask)
        return question_output
    
        #def episodic_memory_module(self):
        
        epm_output = EpisodicMemoryModule(units=self.hidden_units, memory_steps=self.num_memory_passes,
                                  emb_dim=self.word_vector_size, batch_size=self.batch_size,
                                  dropout=self.dropout)([facts_tensors, question_output])  
    
        start_idx_probs = Dense(units=self.output_dim, activation="softmax", name="StartIdxProbs",
                            kernel_regularizer=regularizers.l2(self.regularization_val))(epm_output) 

        end_idx_probs = Dense(units=self.output_dim, activation="softmax", name="EndIdxProbs",
                              kernel_regularizer=regularizers.l2(self.regularization_val))(epm_output)


        # Defining the Model Architecture.
        DMN_model = Model(inputs=[context_input, sent_end_idx_input, question_input],
                          outputs=[start_idx_probs, end_idx_probs])

        DMN_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), 
                                                              metrics=['categorical_accuracy'])        
        if self.summary:
            print(DMN_model.summary())  
            
        return DMN_model  
    
    def get_answer_span(self, answer_start, answer_end):
        assert len(answer_start) == len(answer_end)
        y_answer_start, y_answer_end= ([] , [])
        start_arr, end_arr = (np.zeros(shape=(self.output_dim,), dtype=float),
                              np.zeros(shape=(self.output_dim,), dtype=float))        
        for sample_idx in range(len(answer_start)): 
            start_arr[answer_start[sample_idx]] = 1.0
            end_arr[answer_end[sample_idx]] = 1.0

            y_answer_start.append(start_arr)
            y_answer_end.append(end_arr)
        return (np.array(y_answer_start), np.array(y_answer_end))
