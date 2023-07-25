# download fastText pretrained vectors english  (~6GB)

mkdir ./data/fastText/
curl -o ./data/fastText/cc.en.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
cd ./data/fastText/
gzip -d cc.en.300.bin.gz 
cd ../../

cd models
curl -o llama-2-13b-chat.ggmlv3.q4_1.bin https://media.quinnpatwardhan.com/data/llama-2-13b-chat.ggmlv3.q4_1.bin
cd ..