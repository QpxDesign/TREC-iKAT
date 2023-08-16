from pyserini.search.lucene import LuceneSearcher
from utils.classify_passages import determinePassageReliability
import json

searcher = LuceneSearcher('data/clueweb/indexes/ikat_collection_2023_02') #indexes/ikat_collection_2023_01

def getPassagesFromSearchQuery(query, maxNumberPassages=10):
    hits = searcher.search(q=query,k=maxNumberPassages)
    return filterOutUnreliablePassages(hits)

def filterOutUnreliablePassages(passages):
    final = []
    for passage in passages:
        predicted_label = determinePassageReliability(json.loads(passage.raw)["contents"])
        if predicted_label == 'reliable':
            final.append(passage)
    return final

"""
a = getPassagesFromSearchQuery("Can i eat fish on a keto diet?",25)
b = json.loads(a[0].raw)["contents"]
print(b)
"""