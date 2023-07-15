## TREC-iKAT

[Official Page](https://www.trecikat.com/)

### Resources

[FastText](https://github.com/facebookresearch/fastText) - a library for efficient learning of word representations and sentence classification.
[Yake/Yet Another Keyword Extractor](https://github.com/LIAAD/yake) - a light-weight unsupervised automatic keyword extraction method which rests on text statistical features extracted from single documents to select the most important keywords of a text

### Notes:

run `bash install-data.sh` to install fastText Data (~5 GB)

### Pipeline for Determining PTKB-Passage Simulatory

[Find Keywords in PTKB & User Input] → [determine cosine simulatory between keywords] → [rank passages by simulatory]
