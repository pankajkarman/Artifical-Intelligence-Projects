import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

class PreProcess():
    def __init__(self, trainfile, testfile, glovefile):
        self.trainfile = trainfile
        self.testfile  = testfile
        self.glovefile = glovefile
    
    def get_data(self, train=True):
        if train:
            train_df_all = pd.read_json(self.trainfile).reset_index(drop=True)
        else:
            train_df_all = pd.read_json(self.testfile).reset_index(drop=True)  
            
        train_df = train_df_all[train_df_all.is_impossible==False][["context", "question", "answer_text",
                                                            "answer_start", "title"]].reset_index(drop=True)
        train_df.answer_start = train_df.answer_start.astype(int)
        contexts, questions, answers, answer_start = (train_df.context.values,
                                                      train_df.question.values, 
                                                      train_df.answer_text.values,
                                                      train_df.answer_start.values)
        answer_end = np.array([answer_start[idx] + len(answers[idx]) for idx in range(len(answer_start))])
        return contexts, questions, answers, answer_start, answer_end

    def get_word_vector_dict(self):
        with open(self.glovefile) as glove_text:
            word_embeddings = [line.split(" ") for line in glove_text.readlines()]
        word_vector_dict = {element[0]:list(map(float, element[1:])) for element in word_embeddings}    
        return word_vector_dict    
    
    @staticmethod
    def tokenize(string):
        tokens = [token.replace("``", '"').replace("''", '"').lower() for token in word_tokenize(string)]
        split_tokens = []
        for token in tokens:
            split_tokens.extend(re.split('(\W+)', token))
        return [token for token in split_tokens if token!=" " and token!=""]

    def get_embedding(self, tokens):
        word_vector_dict = self.get_word_vector_dict()
        tokens = np.array(tokens)
        embedding  = []
        for token in tokens:
            if token in word_vector_dict.keys(): 
                embedding.extend(word_vector_dict[token])
            else:
                # Words with no embedding are assigned the 'unk' token vectorization (already in GloVe)
                embedding.extend(word_vector_dict["unk"])

        return np.array(embedding)

    @staticmethod
    def get_sent_end_idx(context_tokenizations):
        return np.array([np.where(np.array(context)==".")[0] for context in context_tokenizations])

    def get_padded_inputs(self, tokenized_inputs, string_type="context"):
        assert isinstance(tokenized_inputs[0], list)==True # Assert multiple samples
        embedded_inputs = [self.get_embedding(tokenized_input) for tokenized_input in tokenized_inputs]

        if string_type=="context":
            padded_input = pad_sequences(embedded_inputs, max_context_len*word_vector_size, padding="post",
                                    dtype="float32").reshape(len(tokenized_inputs), -1, word_vector_size)
        elif string_type=="question":
            padded_input = pad_sequences(embedded_inputs, max_question_len*word_vector_size, padding="post",
                                    dtype="float32").reshape(len(tokenized_inputs), -1, word_vector_size)
        else:
            print("Incorrect string_type parameter value.")

        return padded_input  
    
    def process(self, contexts, questions, answers):
        tokenized_contexts = np.array([self.tokenize(context) for context in contexts])
        tokenized_questions = np.array([self.tokenize(question) for question in questions])
        tokenized_answers = np.array([self.tokenize(answer) for answer in answers])
        sent_end_indices = self.get_sent_end_idx(tokenized_contexts)
        return tokenized_contexts, tokenized_questions, tokenized_answers, sent_end_indices
