# download fastText pretrained vectors english  (~6GB)

#mkdir ./data/fastText/
#curl -o ./data/fastText/cc.en.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
#cd ./data/fastText/
#gzip -d cc.en.300.bin.gz 
#cd ../../


# download llama 7B chat (~12GB)
mkdir ./data/llama/
curl -o ./data/llama/llama-2-7b-chat.zip https://media.quinnpatwardhan.com/data/llama-2-7b-chat.zip
cd ./data/llama/zx
unzip ./data/llama-2-7b-chat.zip -d llama-2-7b-chat