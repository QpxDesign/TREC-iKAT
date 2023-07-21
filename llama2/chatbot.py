from typing import Optional
import json
import fire
from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    covno_history = []
    while True:
        i = input("USER: ")
        covno_history.append([{"role": "user", "content":i}])
    
        results = generator.chat_completion(
            covno_history,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )
        print(results)
        print(results[-1]['generation']['content'])

    
if __name__ == "__main__":
    fire.Fire(main)
