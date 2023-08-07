from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('data/clueweb/indexes/ikat_collection_2023_02') #indexes/ikat_collection_2023_01

def getPassagesFromSearchQuery(query):
    hits = searcher.search(query)
    return hits