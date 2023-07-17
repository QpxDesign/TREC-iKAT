from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2-medium')
set_seed(42)

class Chat:
    def __init__(self):
        self.data = []
    def runGen(self,input):
        gens = generator(input, max_length=30, num_return_sequences=5)
        for g in gens:
            print(g['generated_text'])


x = Chat()
x.runGen("Saturday Night Live")