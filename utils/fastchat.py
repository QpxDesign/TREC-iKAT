from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.rank_passage_sentances import rank
from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)
import torch
import time

# SET THIS (NEED A LOT OF VRAM IF USING LLAMA WITH GPU AND FASTCHAT WITH GPU)
USE_GPU = False

device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'


tokenizer = AutoTokenizer.from_pretrained(
    "lmsys/fastchat-t5-3b-v1.0", legacy=False)
# ONLY HAVE load_in_8bit True if using GPU with limited VRAM
if USE_GPU:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "lmsys/fastchat-t5-3b-v1.0", load_in_8bit=True)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")


def summarize_with_fastchat(passage, question):
    MAX_PASSAGE_LENGTH = 512  # char
    # Ensure the task is properly formatted
    passage = f"Human: Summarize this passage in 1-2 sentences {rank(passage, question)[:MAX_PASSAGE_LENGTH]} Assistant: "
    encoded_input = tokenizer.encode(passage, return_tensors='pt').to(device)
    output = model.generate(
        encoded_input, max_length=1024, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ', '')
    decoded_output = decoded_output.replace("  ", " ")
    print("GENERATED FASTCHAT SUMMARY")
    return decoded_output


"""
start_time = time.time()
a = summarize_with_fastchat(
    "test passage"
    , "who is John J DeGioia?")

print(a)
print(f"GENERATED SUMMARY in {time.time()-start_time}sec")
"""
