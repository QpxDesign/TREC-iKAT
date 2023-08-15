from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.rank_passage_sentances import rank
from transformers import (   
    AutoModelForSeq2SeqLM,   
    T5Tokenizer,
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

def summarize_with_fastchat(passage,question):
    MAX_PASSAGE_LENGTH = 256 #char
    passage = f"Human: Summarize this passage in 1-2 sentences {rank(passage, question)[:MAX_PASSAGE_LENGTH]} Assistant: "  # Ensure the task is properly formatted
    print("GENERATING FASTCHAT SUMMARY")
    encoded_input = tokenizer.encode(passage, return_tensors='pt')
    output = model.generate(encoded_input, max_length=512, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ','')
    decoded_output = decoded_output.replace("  "," ")
    print("GENERATED FASTCHAT SUMMARY")
    return decoded_output
