from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher('data/clueweb/indexes/ikat_collection_2023_02') #indexes/ikat_collection_2023_01

def getPassagesFromSearchQuery(query):
    hits = searcher.search(query)
    return hits

a = getPassagesFromSearchQuery("What are some good colleges for computer science in the netherlands?")[0]
b = json.loads(a.raw)["contents"]
print(b)