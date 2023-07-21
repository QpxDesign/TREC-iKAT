from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2-large')
set_seed(42)

class Chat:
    def __init__(self):
        self.data = []
    def runGen(self,input):
        return generator(input, max_length=30, num_return_sequences=5)


x = Chat()
print(x.runGen("the best university in the netherlands for computer science is"))