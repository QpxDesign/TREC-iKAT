## TREC-iKAT

[Official Page](https://www.trecikat.com/)

### Resources

- [FastChat T5](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) - large language model thats much better at summarization than LLAMA2
- [Sentance-Transformers](https://www.sbert.net/) - a Python framework for state-of-the-art sentence, text and image embeddings, based on BERT
- [Pyserini](https://github.com/castorini/pyserini) - a Python toolkit for reproducible information retrieval research with sparse and dense representations (currently using to get passages from query, using BM25-based searcher)
- [LLAMA2](https://github.com/facebookresearch/llama) - large language model with a built-in chat model, on-par with ChatGPT (using 7B params chat rn)
- [WikiText](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) - The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License. Produced by Cloudflare's Einstien AI Research Lab, we're using it to train text classifer on what kind of passages to look for
- [scikit-learn](https://scikit-learn.org/stable/index.html) - toolkit for AI based classification, analysis, and more, currently we're using it to determine reliability of passages

### Notes:

install pip dependencies: `pip install -r requirements.txt` [MacOS instructions of llama.cpp python](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

run `bash scripts/install-data.sh` to install llama model (13B-Chat) from my server (~8 GB)

run `python3 -m fastchat.serve.cli --model-path lmsys/fastchat-t5-3b-v1.0` to install fastchat-t5 model (~7 GB) ~~(CURRENTLY NOT USED)~~

running ptkb_similarity or rank_passage_sentances (both are run in full-run) will download several BERT models (a few GB total)

you should be able to run this on any basically any CPU/OS/Architecture/With or without GPU

set PYTHON_PATH variable: `export PYTHONPATH=$PWD:$PYTHONPATH`

install wikitext-103 from [Cloudflare Einstien](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) (~600MB Uncompressed), unzip it, and drag it into data/text-classification (used to train passage classifier)

install news articles corpus from [Kaggle](https://www.kaggle.com/datasets/sbhatti/news-articles-corpus?resource=download) (~2GB) and place it in data/news-articles (used to train passage classifier)

if you're using ChatGPT instead of Llama, [generate an OPENAI API Key](https://openai.com/blog/openai-api), create a .env in the main dir, and put your key in it: `OPENAI_API_KEY=<KEY GOES HERE>`
please note that a full-run using ChatGPT may use ~$1-3 of credit

#### Running Pyserini with clueweb22

1. Place ikat collections (named `ikat_collection_2023_0n.json`) into /data/clueweb/
2. Format by running `bash scripts/format_ikat_collection.sh` (this will take a long time)
3. Generate the index: `python -m pyserini.index.lucene --collection JsonCollection  --input ~/TREC-iKAT/data/clueweb --index indexes/ikat_collection_2023  --generator DefaultLuceneDocumentGenerator  --threads 1  --storePositions --storeDocvectors --storeRaw`

#### Helpful Commands

Format output JSON `cat output/AUG17_RUN_2.json | python -m json.tool > output/AUG17_RUN_2_F.json`
