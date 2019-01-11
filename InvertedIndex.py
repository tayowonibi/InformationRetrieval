import pandas as pd
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import math
import numpy as np

from Helper import *


NUM_TOPICS = 50
WORD2VEC_FILE ="C:\\Users\\mowonibi\\Downloads\\crawl-300d-2M.vec"

class Inverted_Index:
    """indexer class for managing the index i.e. adding index, removing index etc"""
    
    def __init__(self, file_location="products/products.csv",field_names={"category" :helper.simple_spliter_tokenizer ,"title":helper.custom_tokenizer,"description":helper.custom_tokenizer}):
        self.file_location = file_location #file to index
        self.field_names  = field_names #This contains the set of field names used in the index and their tokenizer
        self.term_frequency =defaultdict( lambda: defaultdict(lambda: defaultdict(int)) ) # maps/dict of dict/maps of field -> term ->docs-> freq of occurence in doc
        self.document_tokens = defaultdict( lambda: defaultdict(list) ) #field  ->docs ->terms
        self.idf =defaultdict (lambda: dict() )#idf of field-> term-> idf
        self.inverted_index_term_proximity = defaultdict( lambda:defaultdict(lambda: defaultdict(list))) #field ->term ->doc ->list of positions in document
        self.corpus =None #pandas dataframe of the products 
        self.document_length=defaultdict( lambda:defaultdict(lambda: defaultdict(int))) #field -> document_id -> length of document
        self.num_of_doc=0 #total number of doc /products
        self.avg_doc_len = dict() #field -> average document length
        self.avg_idf = dict() #field -> avg idf
        self.field_dict=dict()
        self.field_corpus=dict()
        self.field_model=dict()
        self.term_embeddings= dict() #term-> embedding  #defaultdict(lambda: np.zeros((300,))) 
        self.doc_avg_embeddings=defaultdict(lambda: defaultdict(lambda: np.zeros((300,)))) # field ->doc ->embedding
    
    def index_texts(self, file_location=None, include_doc2vec=False, index_doc_embedding=False):
        """indexs the files in the text"""
        sum_of_doc_len=defaultdict(int) #field ->sum of document lenght  which can be used to compute average document length lateron
        field_doc_tokens=defaultdict(lambda: defaultdict(list)) #field->doc->tokens
        if file_location is None:
            file_location = self.file_location
        self.corpus = pd.read_csv(file_location)
        
        for index, row in self.corpus.iterrows(): #loop rows of products
            self.num_of_doc += 1
            #if (self.num_of_doc >20):
            #    break
            for field_name, tokenizer  in self.field_names.items():   #loops each field
                tokenized_text = tokenizer(row[field_name])
                if include_doc2vec :
                    field_doc_tokens[field_name][index]=tokenized_text
                current_doc_len= len(tokenized_text)
                self.document_length[field_name][index] =current_doc_len
                sum_of_doc_len[field_name] +=current_doc_len
                self.document_tokens[field_name][index]=tokenized_text

                for k,word in enumerate(tokenized_text):     #loops each tokenized text
                    self.inverted_index_term_proximity[field_name][word][index].append( k)
                    self.term_frequency[field_name] [word][index] += 1

        sum_of_idf=defaultdict(float) #field -> sum of the idf
        #computes the idf of each term in the corpus
        for field_name, field_tf in self.term_frequency.items(): 
            for word, tf in field_tf.items():
                self.idf[field_name][word] = math.log(self.num_of_doc - len(tf) + 0.5) - math.log(len(tf) + 0.5)
                sum_of_idf[field_name]+=self.idf[field_name][word] #aggregate/sums the idf by summing to the aggregator
        
        for field_name in self.field_names:
            self.avg_doc_len[field_name] = sum_of_doc_len[field_name] / self.num_of_doc
            self.avg_idf[field_name] = sum_of_idf[field_name] / self.num_of_doc
        
        if include_doc2vec :
            self._index_doc2vec(  field_doc_tokens)
        
        if index_doc_embedding:
            self.index_doc_avg_embedding()
    
    def _index_doc2vec(self,  field_doc_tokens):
        #field_corpus =dict()
        #field_dict =dict()
        for field_name, _  in self.field_names.items():
            dictionary = corpora.Dictionary([list(self.term_frequency[field_name].keys())])
            t_documents = [dictionary.doc2bow(text) for  text in  field_doc_tokens[field_name].values()]  #create bog of words (multi-set)
            model = gensim.models.LdaMulticore(t_documents, num_topics = NUM_TOPICS, id2word=dictionary, passes=30, workers=3)
            model.save(field_name +'_model50.gensim')
            self.field_dict[field_name]=dictionary
            self.field_corpus[field_name]=t_documents
            self.field_model[field_name]=model
            
    def index_doc_avg_embedding (self):
        self.load_term_embedding()
        for field_name  in self.field_names.keys():
            for doc_id  in self.document_tokens[field_name].keys():
                self.doc_avg_embeddings[field_name][doc_id] = self.get_avg_doc_embedding(self.document_tokens[field_name][doc_id]) 
                   
                
    def get_avg_doc_embedding (self, tokens):
        embedding_accum = np.zeros((300,))
        d_len=0
        for tkn in tokens:
            if (tkn in self.term_embeddings.keys()):
                embedding_accum +=self.term_embeddings[tkn]
                d_len+=1
        return embedding_accum/d_len

    def  load_term_embedding(self):
        term_set= set()
        for field_name in self.field_names.keys():
            term_set = set.union(term_set,self.term_frequency[field_name].keys() )
        #print(term_set)
        with open(WORD2VEC_FILE, encoding='utf8') as f:
            next(f)
            for l in f:
                w = l.split(' ')
                if w[0] in term_set:
                    self.term_embeddings[w[0]] = np.array([float(x) for x in w[1:301]]) 