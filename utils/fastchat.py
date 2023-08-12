from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import (   
    AutoModelForSeq2SeqLM,   
    T5Tokenizer,
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

def summerize_with_fastchat(passage):
    MAX_PASSAGE_LENGTH = 2040 #char
    passage = f"Human: Using this passage, awnser this question: I live in the netherlands and I can't tolerate cold temperatures. What are some good colleges? {passage[:MAX_PASSAGE_LENGTH]} Assistant: "  # Ensure the task is properly formatted
    encoded_input = tokenizer.encode(passage, return_tensors='pt')
    output = model.generate(encoded_input, max_length=512, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ','')
    decoded_output = decoded_output.replace("  "," ")
    return decoded_output
