from collections import defaultdict
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from enum import Enum
import math

from sklearn.metrics.pairwise import cosine_similarity

from Helper import *
from RankFunction import *
from InvertedIndex import *

class Searcher:
    """class for searching document based on a given index"""
    def __init__(self, doc_index ):
        self.doc_index = doc_index
        
    def field_term_search(self, field_name, query_tokenized_terms_list, max_separation_between_terms = None, q_mode=QueryTermMode.OR) : #max_separation_between_terms removes effect of q_mode
        """
        search list of tokens, in a specific field, given query mode and max separation between terms in a given document
        Returns list of set of doucment/product id
        """
        
        result_set =None
        prev_result_set=None
        if max_separation_between_terms is None :
            for q_term in query_tokenized_terms_list:
                result_set =set(self.doc_index.inverted_index_term_proximity[field_name][q_term].keys()) if result_set is None else q_mode(result_set, set(self.doc_index.inverted_index_term_proximity[field_name][q_term].keys()))
        else :
            for q_term in tokenized_terms_list:
                if result_set is None :
                    prev_result_set =self.doc_index.inverted_index_term_proximity[field_name][q_term]
                    result_set = set(self.doc_index.inverted_index_term_proximity[field_name][q_term].keys())
                else :
                    temp_result_set = set()
                    curr_result_set = self.doc_index.inverted_index_term_proximity[field_name][q_term]
                    for doc_id in QueryTermMode.AND(result_set, set(curr_result_set.keys())):
                        positions_current_term =curr_result_set[doc_id]
                        positions_previous_term =prev_result_set[doc_id]
                        for tc_position in positions_current_term :
                            for tp_position in positions_previous_term:
                                tp_tc =tc_position - tp_position
                                if (tp_tc>0 and tp_tc<=max_separation_between_terms ):
                                    temp_result_set.add(doc_id)
                                    break
                    result_set =temp_result_set
                    prev_result_set = curr_result_set             
        return result_set

    
    def term_search(self, query_terms_as_string, q_mode=QueryTermMode.OR):
        """
        search for query string in all fields
        Returns set of document/product ids 
        """
        result_set=None
        for field_name, tokenizer  in self.doc_index.field_names.items():   
            tokenized_text = tokenizer(query_terms_as_string)
            result_set   = self.field_term_search (field_name, tokenized_text, q_mode=q_mode) if result_set is None else QueryTermMode.OR(result_set, self.field_term_search (field_name, tokenized_text, q_mode=q_mode))   
        return result_set
    
    def ranked_search(self, field_name, query_terms_as_string, length, ranking_function):
        """Returns the `size` most relevant documents based on the `query`"""
        scores = ranking_function(field_name, query_terms_as_string)
        scores=sorted(scores,  key = lambda x: (-scores[x], x))
        #indexes= list(scores.keys())
        return scores[:length]
    
    