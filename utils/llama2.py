from llama_cpp import Llama
import sys
#llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin", n_ctx=10240, n_threads=16) - USE IF NO GPU/NOT ENOUGH VRAM
#llm = Llama(model_path="./models/llama-2-70b-chat.ggmlv3.q2_K.bin", n_gqa=8, n_ctx=10240, n_threads=16) - USE WITH NO GPU (SEE README) OR IF YOU HAVE LOADS OF VRAM
llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q3_K_M.bin", n_ctx=10240, n_gpu_layers=50) # NEEDS ~8GB+ OF VRAM, MUCH FASTER
from utils.rank_passage_sentances import rank
from utils.prevent_trail_off import prevent_trail_off
import time
import re
from utils.remove_extra_spaces import remove_extra_spaces

def gen_response(prompt,previous_chats):
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    print(f'generating response: {full_prompt}')
    output = llm(full_prompt,max_tokens=256, stop=["Q:"], echo=True) 
    print(output)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    return ans

def gen_summary(passage,question):
    MAX_PASSAGE_LENGTH = 512 #num characters
    prompt = f"Q: Summarize this passage in 1 sentence: {rank(passage, question)[:MAX_PASSAGE_LENGTH]} A: "
    output = llm(prompt,max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace("Sure! Here is a summary of the passage in one sentence:",'')
    ans = ans.replace('Here is a summary of the passage in one sentence:','')
    ans = ans.replace('Here is the summary in one sentence based on the information from the passages:','')
    return ans

def answer_question_from_passage(passage,question,previous_chats):
    prompt = f"Answer this question - {question} - using this information - {remove_extra_spaces(passage)}."
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt,max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace("Sure! Here is a summary of the passage in one sentence:",'')
    ans = ans.replace('Here is a summary of the passage in one sentence:','')
    ans = ans.replace('Here is the summary in one sentence based on the information from the passages:','')
    ans = ans.replace('Sure! Based on the information provided, here is a summary of the passage in one sentence:','')
    ans = ans.replace("\n","")
    return prevent_trail_off(ans)
"""
start_time = time.time()
full_prompt = "Q: Answer this question -  can't exercise too much because of the heart problem that I have and Because of my kidney problem, I have to drink water frequently to stay hydrated. Can you help me find a diet for myself considering that I'm vegetarian, allergic to soybeans, lactose intolerant, can't exercise too much, and should drink water regularly? - using this information - The passage discusses lactose intolerance, dairy products, and diet. Some people drink lactaid milk and have yogurt regularly, while others ask for alternatives to cheese and cream with meals. Some people find that their diet is healthier now that they can't have ice-cream, cheese cakes, or puddings. Others have questions about lactose intolerance and what to take in a ruotine diet.\n Children between the ages of one and five should consume no more than three eight-ounce glasses of milk per day. Drinking about one-and-a-half cups of milk per day is enough to meet the calcium needs of children between ages one and three. Lactose intolerance occurs when the body slowly breaks down lactose, which is the sugar in milk. It can be uncomfortable for some children, but it is not dangerous.\n Stay physically fit, drink plenty of water, avoid foods that cause a sticky stool, and avoid overdosing on sugar. A: "
output = llm(full_prompt,max_tokens=256, stop=["Q:"], echo=True) 
print(output)
print(f"GENERATED OUTPUT in {time.time()-start_time} seconds")
"""