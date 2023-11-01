from pyserini.search.lucene import LuceneSearcher
import json

# indexes/ikat_collection_2023_01
searcher = LuceneSearcher('data/passage-index')


def getPassagesFromSearchQuery(query, maxNumberPassages=10, exactMatchRequired=False):
    hits = searcher.search(q=query, k=maxNumberPassages)
    if exactMatchRequired:
        matches = []
        for h in hits:
            if query.lower() in json.loads(h.raw)["contents"].lower():
                matches.append(h)
        return matches
    return hits


"""
a = getPassagesFromSearchQuery(
    "Sure! What kind of car are you looking for? Would you like something sleek and sporty or practical and reliable?", 100)
b = json.loads(a[2].raw)["contents"]
print(b)
"""
