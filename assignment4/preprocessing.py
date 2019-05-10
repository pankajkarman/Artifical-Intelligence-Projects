import nltk, numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def binary_label(x):
    nrows = x.shape[0]
    trans = np.zeros((nrows, 2))
    
    for i, value in enumerate(x):
        if value == 0:
            trans[i, 0] = 1
        else:
            trans[i, 1] = 1
    return trans

class PreProcess():
    def __init__(self, infile):
        self.infile = infile
        
    def OneHotEncoder(self, x):
        spam = {'ham':0, 'spam':1}
        return spam[x]
    
    def process(self, test_size=0.2):
        df = pd.read_csv(self.infile, sep="\t", header=None)
        y_input, X_input = df[0].astype(str), df[1].astype(str)
        ylabels = np.array([self.OneHotEncoder(out) for out in y_input]).reshape(-1,1)
        
        vectorizer = TfidfVectorizer(tokenizer = self.stemmed_words,\
                                     stop_words = stopwords.words('english'))
        vectorizer.fit(X_input)
        Xvector = vectorizer.transform(X_input).toarray()
        X_train, X_test, y_train, y_test = train_test_split(Xvector, \
                                                            ylabels, test_size=test_size)
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def stemmed_words(doc):
        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in word_tokenize(doc)] 
