# download fastText pretrained vectors english
curl -o ./data/fastText/cc.en.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
mkdir ./data/fastText/
cd ./data/fastText/
gzip -d cc.en.300.bin.gz 
