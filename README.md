## TREC-iKAT

[Official Page](https://www.trecikat.com/)

### Resources

- [FastText](https://github.com/facebookresearch/fastText) - a library for efficient learning of word representations and sentence classification.
- [Yake/Yet Another Keyword Extractor](https://github.com/LIAAD/yake) - a light-weight unsupervised automatic keyword extraction method which rests on text statistical features extracted from single documents to select the most important keywords of a text
- [GPT-2 Large](https://huggingface.co/gpt2-large2) - Pretrained model on English language using a causal language modeling (CLM) objective, has autocomplete capabilities
- [DQN TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v1 task from Gymnasium.
- [LLAMA2](https://github.com/facebookresearch/llama) - large language model with a built-in chat model, on-par with ChatGPT (using 7B params chat rn)

### Notes:

install pip dependencies: `pip install -r requirements.txt` [MacOS instructions of llama.cpp python](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

run `bash install-data.sh` to install llama model (13B-Chat) from my server (~8 GB)

running ptkb_similarity will download several BERT models (a few GB total)

you should be able to run this on any basically any CPU/OS/Architecture/With or without GPU

#### Running Pyserini with clueweb22
1. Place ikat collections (named `ikat_collection_2023_0n.json`) into /data/clueweb/
2. Format by running `bash format_ikat_collection.sh` (this will take a long time)
3. Generate the index: `python -m pyserini.index.lucene --collection JsonCollection  --input ~/TREC-iKAT/data/clueweb --index indexes/ikat_collection_2023_02  --generator DefaultLuceneDocumentGenerator  --threads 1  --storePositions --storeDocvectors --storeRaw`