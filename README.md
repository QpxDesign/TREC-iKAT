## TREC-iKAT

[Official Page](https://www.trecikat.com/)

### Resources

[FastText](https://github.com/facebookresearch/fastText) - a library for efficient learning of word representations and sentence classification.
[Yake/Yet Another Keyword Extractor](https://github.com/LIAAD/yake) - a light-weight unsupervised automatic keyword extraction method which rests on text statistical features extracted from single documents to select the most important keywords of a text

### Notes:

run `bash install-data.sh` to install fastText Data (~5 GB)

see a sample output for the releveance algorthim [here](https://docs.google.com/spreadsheets/d/1-VU4-3qC3Q7mTdF9iZUgs3RjQVbYGD9ixf2JdMcMJRc/edit?usp=sharing)

### Pipeline for Determining PTKB-Passage Simulatory

[Find Keywords in PTKB & Response] → [determine cosine simulatory between keywords] → [rank passages by simulatory]
