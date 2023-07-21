# download fastText pretrained vectors english  (~6GB)

#mkdir ./data/fastText/
#curl -o ./data/fastText/cc.en.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
#cd ./data/fastText/
#gzip -d cc.en.300.bin.gz 
#cd ../../

# download llama 7B chat (~12GB)
cd ./llama
curl -o ./llama-2-7b-chat.zip https://media.quinnpatwardhan.com/data/llama-2-7b-chat.zip
unzip ./data/llama-2-7b-chat.zip