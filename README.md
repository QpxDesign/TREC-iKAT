## TREC-iKAT

[Official Page](https://www.trecikat.com/)

### Resources

- [FastText](https://github.com/facebookresearch/fastText) - a library for efficient learning of word representations and sentence classification.
- [Yake/Yet Another Keyword Extractor](https://github.com/LIAAD/yake) - a light-weight unsupervised automatic keyword extraction method which rests on text statistical features extracted from single documents to select the most important keywords of a text
- [GPT-2 Large](https://huggingface.co/gpt2-large2) - Pretrained model on English language using a causal language modeling (CLM) objective, has autocomplete capabilities
- [DQN TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v1 task from Gymnasium.

### Notes:

run `bash install-data.sh` to install fastText Data (~5 GB)

see a sample output for the releveance algorthim [here](https://docs.google.com/spreadsheets/d/1-VU4-3qC3Q7mTdF9iZUgs3RjQVbYGD9ixf2JdMcMJRc/edit?usp=sharing)

### Pipeline for Determining PTKB-Passage Simulatory

[Find Keywords in PTKB & Response] → [determine cosine similarity between keywords] → [rank passages by similarity]

### Pipeline for GPT Generation

_[user input]_ -> **[derive auto-completable string]** -> [generate sequences] -> [score w/ Q function] -> [return a higher-scoring sequences]

_italics_ = implemented \
**bold** = working on

### Pipeline for DQN

How to use a DQN (Deep-Q Network): The neural network takes the current state of the environment as input and outputs a Q-value for each possible action. The agent then selects the action with the highest Q-value to maximize its rewards.
