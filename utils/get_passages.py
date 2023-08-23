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


a = getPassagesFromSearchQuery("
 Finger limes and black garlic are two very different ingredients, each with their own unique flavors and textures. Here are some key differences to help you decide which one is best for your cooking needs:

Flavor:

* Finger limes have a bright, citrusy flavor that is often described as a cross between a lime and a lemon. The juice of the fruit is tart and slightly sweet, with a hint of floral notes.
* Black garlic has a rich, savory flavor that is sweet and caramelized, similar to roasted garlic but more intense. It has a deep, dark flavor that is often used to add depth to dishes.

Texture:

* Finger limes have a juicy, slightly crunchy texture that is similar to a citrus fruit. The pulp of the fruit is tart and slightly grainy, while the skin is thin and easy to peel.
* Black garlic has a soft, creamy texture that is chewy and sticky. It can be chopped or minced to add texture to dishes.

Preparation:

* Finger
",100,True)
b = json.loads(a[0].raw)["contents"]
print(b)
"""