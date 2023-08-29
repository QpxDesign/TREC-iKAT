from utils.remove_extra_spaces import remove_extra_spaces
import torch
import json
import time
# from utils.prevent_trail_off import prevent_trail_off
from utils.rank_passage_sentances import rank
from llama_cpp import Llama
llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin",
            n_ctx=10240, n_threads=16)  # - USE IF NO GPU/NOT ENOUGH VRAM
# llm = Llama(model_path="./models/llama-2-70b-chat.ggmlv3.q2_K.bin", n_gqa=8, n_ctx=10240, n_threads=16) - USE WITH NO GPU (SEE README) OR IF YOU HAVE LOADS OF VRAM
# llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q3_K_M.bin",
# n_ctx=10240, n_gpu_layers=50)  # NEEDS ~8GB+ OF VRAM, MUCH FASTER

torch.cuda.empty_cache()


def gen_response(prompt, previous_chats):
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['utterance']} A: {pc['response']}"

    # When answering, always include all necessary information, repeating anything from the previous question(s) when necessary.
    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt, max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    return ans


def gen_summary(passage, question):
    MAX_PASSAGE_LENGTH = 512  # num characters
    prompt = f"Q: Summarize this passage in 1 sentence: {rank(passage, question)[:MAX_PASSAGE_LENGTH]} A: "
    output = llm(prompt, max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace(
        "Sure! Here is a summary of the passage in one sentence:", '')
    ans = ans.replace('Here is a summary of the passage in one sentence:', '')
    ans = ans.replace(
        'Here is the summary in one sentence based on the information from the passages:', '')
    return ans


def answer_question_from_passage(passage, question, previous_chats):
    prompt = f"Answer this question - {question} - using this information - {remove_extra_spaces(passage)} "
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt, max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace(
        "Sure! Here is a summary of the passage in one sentence:", '')
    ans = ans.replace('Here is a summary of the passage in one sentence:', '')
    ans = ans.replace(
        'Here is the summary in one sentence based on the information from the passages:', '')
    ans = ans.replace(
        'Sure! Based on the information provided, here is a summary of the passage in one sentence:', '')
    ans = ans.replace("\n", "")
    return ans


def determine_passage_relevance(passage, statement):
    prompt = f"Q: Is this passage - {json.loads(passage.raw)['contents']} relevant to this response - {statement}? Answer with either 'yes' or 'no'. A: "
    output = llm(prompt, max_tokens=256, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = remove_extra_spaces(ans)
    print(ans)
    if ans[0:2].lower() == "no":
        return False
    if ans[0:3].lower() == 'yes':
        return True
    return False


"""
start_time = time.time()
full_prompt = "Q: List the names of 5 good diets to use to loose weight. A: 1. DASH Diet, 2. Keto, 3. Mediterranean. Q: Can i eat fish in any of those diets? A:"
output = llm(full_prompt, max_tokens=1024, stop=["Q:"], echo=True)
print(output)
ans = output["choices"][-1]["text"]
ans = ans.split(" A: ")[-1]
print(ans)
print(f"GENERATED OUTPUT in {time.time()-start_time} seconds")
"""
