from nltk.tokenize import TweetTokenizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from enum import Enum
import math



class QueryTermMode(Enum): 
    """ 
    Enum Class :
    Enumeration of functions to to use for the mode in combining the results of each query term
    """
    OR = set.union # all documents containing any of the query terms should be returned
    AND = set.intersection #only documents containing all of the query terms to be returned

class Helper:
    """
    Class containing any helper function to be used in the application
    """
    tknzr = TweetTokenizer()
    en_stop = set(nltk.corpus.stopwords.words('english'))
    
	@staticmethod
    def custom_tokenizer( sentence):
        """
        lemmatize lower case tokens of sentence tokenized using nltk tweet tokenizer
        Returns list of tokens
        """
        return [WordNetLemmatizer().lemmatize(token.lower() ) for token in Helper.tknzr.tokenize(sentence) if token not in Helper.en_stop and len(token) > 1]
    
    @staticmethod
    def simple_spliter_tokenizer(sentence, splitter="/"):
        """
        simple splitter based on user-defined regex expressions
        Returns list of tokens
        """
        return [token.lower()  for token in sentence.split(splitter)]