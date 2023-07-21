from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from pyserini.search.hybrid import HybridSearcher

ssearcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
dsearcher = FaissSearcher.from_prebuilt_index(
    'msmarco-passage-tct_colbert-hnsw',
    encoder
)
hsearcher = HybridSearcher(dsearcher, ssearcher)
hits = hsearcher.search('what is a lobster roll')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')