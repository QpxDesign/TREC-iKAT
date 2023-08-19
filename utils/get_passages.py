from pyserini.search.lucene import LuceneSearcher
from utils.classify_passages import determinePassageReliability
import json

searcher = LuceneSearcher('data/passage-index') #indexes/ikat_collection_2023_01

def getPassagesFromSearchQuery(query, maxNumberPassages=10, onlyGoodPassages=True):
    hits = searcher.search(q=query,k=maxNumberPassages)
    if onlyGoodPassages:
        return filterOutUnreliablePassages(hits)
    return hits

def filterOutUnreliablePassages(passages):
    final = []
    for passage in passages:
        predicted_label = determinePassageReliability(json.loads(passage.raw)["contents"])
        if predicted_label == 'reliable':
            final.append(passage)
    return final


"""
a = getPassagesFromSearchQuery("Sous vide cooking technique",100,True)
b = json.loads(a[0].raw)["contents"]
print(b)
"""