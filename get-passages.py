from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/ikat_collection_2023_01') #indexes/ikat_collection_2023_01

def getPassagesFromSearchQuery(query):
    hits = searcher.search(query)
    return hits