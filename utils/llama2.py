from llama_cpp import Llama
llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin", n_ctx=10240)

def gen_response(prompt,previous_chats):
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt,max_tokens=10000, stop=["Q:", "\n"], echo=True) # 250 tokens is TREC iKAT limit
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    return ans

