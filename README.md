## TREC-iKAT

[Official Page](https://www.trecikat.com/)

#### Resources

- [FastChat T5](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) - large language model thats much better at summarization than LLAMA2
- [Sentance-Transformers](https://www.sbert.net/) - a Python framework for state-of-the-art sentence, text and image embeddings, based on BERT
- [Pyserini](https://github.com/castorini/pyserini) - a Python toolkit for reproducible information retrieval research with sparse and dense representations (currently using to get passages from query, using BM25-based searcher)
- [LLAMA2](https://github.com/facebookresearch/llama) - large language model with a built-in chat model, on-par with ChatGPT (using 7B params chat rn)
- [WikiText](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) - The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License. Produced by Cloudflare's Einstien AI Research Lab, we're using it to train text classifer on what kind of passages to look for
- [scikit-learn](https://scikit-learn.org/stable/index.html) - toolkit for AI based classification, analysis, and more, currently we're using it to determine reliability of passages

#### Notes:

install pip dependencies: `pip install -r requirements.txt` [MacOS instructions of llama.cpp python](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

run `bash scripts/install-data.sh` to install llama model (13B-Chat) from my server (~8 GB)

run `python3 -m fastchat.serve.cli --model-path lmsys/fastchat-t5-3b-v1.0` to install fastchat-t5 model (~7 GB)

running ptkb_similarity or rank_passage_sentances (both are run in full-run) will download several BERT models (a few GB total)

set PYTHON_PATH variable: `export PYTHONPATH=$PWD:$PYTHONPATH`

install wikitext-103 from [Cloudflare Einstien](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) (~600MB Uncompressed), unzip it, and drag it into data/text-classification (used to train passage classifier)

install news articles corpus from [Kaggle](https://www.kaggle.com/datasets/sbhatti/news-articles-corpus?resource=download) (~2GB) and place it in data/news-articles (used to train 'less strict' passage classifier)

if you're using ChatGPT instead of Llama, [generate an OPENAI API Key](https://openai.com/blog/openai-api), create a .env in the main dir, and put your key in it: `OPENAI_API_KEY=<KEY GOES HERE>`
please note that a full-run using ChatGPT may use ~$1-3 of credit

#### System Requirements

This was tested/developed/ran from a computer running Ubuntu 22.04 with an RTX 3080 (10GB Version), an Intel i7-11700K (16 total CPU threads). Runs took between 3 and 28 hours, depending mostly on which LLM was used to generate the final responses. There are some tweaks you may need to make if you have different hardware:

- Change n_threads in `utils/llama2.py` to however many CPU threads you have (if you're running on CPU)
  - remove n_gpu_layers=30 if you're not running on GPU
- change references to `"device=cuda"` in `utils/ptkb_similarity.py` and `utils/rank_passage_sentences.py`
- switch which model of LLaMa you're running depending on your system's capabilities:
  - Use quantized versions of LLaMa2, and, if you want to run on GPU or you have limited RAM, make sure you have more ram than the listed RAM usage (if you have too little VRAM to run a model but enough ram, uninstall llama-cpp-python and reinstall without GPU support: `pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir`)
    - [quantized llama2 7B chat](https://huggingface.co/TheBloke/Llama-2-7B-chat-GGML)
    - [quantized llama2 13B chat](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML)
    - [quantized llama2 70B chat](https://huggingface.co/TheBloke/Llama-2-70B-chat-GGML)

#### Running Pyserini with clueweb22

1. Place ikat collections (named `ikat_collection_2023_0n.json`) into /data/clueweb/
2. Format by running `bash scripts/format_ikat_collection.sh` (this will take a long time)
3. Generate the index: `python -m pyserini.index.lucene --collection JsonCollection  --input ~/TREC-iKAT/data/clueweb --index indexes/ikat_collection_2023  --generator DefaultLuceneDocumentGenerator  --threads 1  --storePositions --storeDocvectors --storeRaw`

#### Helpful Commands

Format large output JSON files with `cat output/AUG17_RUN_2.json | python -m json.tool > output/AUG17_RUN_2_F.json`
