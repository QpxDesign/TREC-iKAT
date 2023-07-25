from llama_cpp import Llama
llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin")
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)

output2 = llm(output["choices"][0]["text"] + " Q: Whats the mass of the third planet?")
print(output2)