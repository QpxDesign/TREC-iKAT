from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.search.lucene import LuceneSearcher
import json

splade_searcher = LuceneImpactSearcher(
    './data/passage-index-splade/lucene-index.clueweb-spade-index', './models/distill-splade-max')
bm25_searcher = LuceneSearcher("./data/clueweb22-bm25-index")


def getPassagesFromSearchQuery(query, maxNumberPassages=10):
    hits = splade_searcher.search(q=query, k=maxNumberPassages)
    final_hits = []
    for hit in hits:
        clueweb_docid = spladeDocIdToClueweb(hit.docid)
        doc = getDocFromDocId(clueweb_docid)
        if doc is not None:
            final_hits.append(doc)
    return final_hits


def getDocFromDocId(docid):
    doc = bm25_searcher.doc(docid)
    if doc is None:
        print("could not find doc")
    return doc


def spladeDocIdToClueweb(docid):
    return f"clueweb22-en00{docid[0]}{docid[1]}-{docid[2]}{docid[3]}-{docid[4]}{docid[5]}{docid[6:len(docid)]}"


a = getPassagesFromSearchQuery(
    "Sure! What kind of car are you looking for? Would you like something sleek and sporty or practical and reliable?", 100)
b = json.loads(a[2].raw())["contents"]
print(b)
