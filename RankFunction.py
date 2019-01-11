import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_DIM=300

class RankFunction :
    
    @staticmethod
    def CosineSimilarity(inverted_index, doc_set =None):
        """
        Computes the BM25 scores of document given a query strinng and the query field
        Returns dictionary of document ids and BM25 scores
        """

        def cosine_ranking(field_name, query_terms_as_string):
            docScore  =defaultdict(float)
            tokenized_text = inverted_index.field_names[field_name](query_terms_as_string)
            query_emb =inverted_index.get_avg_doc_embedding(tokenized_text)
            docs_wth_term =set()
            for word in tokenized_text:
                docs_wth_term =set.union(docs_wth_term, inverted_index.term_frequency[field_name][word].keys())
            #print(len(docs_wth_term))
            #print(query_emb)
            for doc_id in docs_wth_term:
                #print(doc_id)
                d_scor = 1- cosine_similarity(query_emb.reshape( 1,EMBEDDING_DIM), inverted_index.doc_avg_embeddings[field_name][doc_id].reshape( 1,EMBEDDING_DIM))[0]
                docScore [doc_id] = d_scor
                #print(doc_id, d_scor)
            return docScore
        return cosine_ranking
    
    
    
    @staticmethod
    def BM25_score(inverted_index, doc_set =None, PARAM_K1=1.2, PARAM_B=0.75, EPSILON=0.25):
        """
        Computes the BM25 scores of document given a query strinng and the query field
        Returns dictionary of document ids and BM25 scores
        """
        #https://github.com/nhirakawa/BM25/blob/master/src/rank.py - consider this implementation
        #https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
        #https://www.quora.com/How-does-BM25-work
        #https://github.com/nhirakawa/BM25/blob/master/src/query.py
        
        def bm25_ranking(field_name, query_terms_as_string):
            docScore  =defaultdict(float)
            tokenized_text = inverted_index.field_names[field_name](query_terms_as_string)
            for word in tokenized_text:
                docs_wth_term =inverted_index.term_frequency[field_name][word]
                for doc, qtf in docs_wth_term.items():
                    #print(self.doc_index.avg_doc_len[field_name], self.doc_index.document_length[field_name][doc])
                    idf = inverted_index.idf[field_name][word] #if self.doc_index.idf[field_name][word] >= 0 else EPSILON * self.doc_index.average_idf[field_name]
                    docScore[doc] += (idf * qtf * (PARAM_K1 + 1)
                      / (qtf + PARAM_K1 * (1 - PARAM_B + PARAM_B * inverted_index.document_length[field_name][doc] / inverted_index.avg_doc_len[field_name])))
            return docScore
        return bm25_ranking