from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher('data/clueweb/indexes/ikat_collection_2023_02') #indexes/ikat_collection_2023_01

def getPassagesFromSearchQuery(query, maxNumberPassages=10):
    hits = searcher.search(q=query,k=maxNumberPassages)
    return hits

"""
a = getPassagesFromSearchQuery("Can i eat fish on a keto diet?")[1]
b = json.loads(a.raw)["contents"]
print(b)
"""