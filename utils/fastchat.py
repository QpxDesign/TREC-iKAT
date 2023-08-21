from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.rank_passage_sentances import rank
from transformers import (   
    AutoModelForSeq2SeqLM,   
    T5Tokenizer,
)
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0",load_in_8bit=True)


def summarize_with_fastchat(passage,question):
    MAX_PASSAGE_LENGTH = 512 #char
    passage = f"Human: Summarize this passage in 1-2 sentences {rank(passage, question)[:MAX_PASSAGE_LENGTH]} Assistant: "  # Ensure the task is properly formatted
    encoded_input = tokenizer.encode(passage, return_tensors='pt').to(device)
    output = model.generate(encoded_input, max_length=1024, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ','')
    decoded_output = decoded_output.replace("  "," ")
    print("GENERATED FASTCHAT SUMMARY")
    return decoded_output


start_time = time.time()
a = summarize_with_fastchat(
    """
n 1975, Georgetown established the Center for Contemporary Arab Studies, soliciting funds from the governments of the United States, Saudi Arabia, Oman, and Libya as well as American corporations with business interests in the Middle East.[49][50] It later returned the money it received from Muammar Qaddafi's Libyan government, which had been used to fund a chair for Hisham Shirabi, and also returned further donations from Iraq.[51] Georgetown ended its bicentennial year of 1989 by electing Leo J. O'Donovan, S.J. as president. He subsequently launched the Third Century Campaign to build the school's endowment.[52] In December 2003, Georgetown completed the campaign after raising over $1 billion for financial aid, academic chair endowment, and new capital projects.[53] In 2005, Georgetown received a $20 million gift from Alwaleed bin Talal bin Abdulaziz Alsaud, member of the Saudi Royal Family; at that time the second-largest donation ever to the university, it was used to expand the activities of the Prince Alwaleed Bin Talal Center for Muslim-Christian Understanding.[54]

In October 2002, Georgetown University began studying the feasibility of opening a campus of the Edmund A. Walsh School of Foreign Service in Qatar, when the non-profit Qatar Foundation first proposed the idea. The School of Foreign Service in Qatar opened in 2005 along with four other U.S. universities in the Education City development. That same year, Georgetown began hosting a two-week workshop at Fudan University's School of International Relations and Public Affairs in Shanghai, China. This later developed into a more formal connection when Georgetown opened a liaison office at Fudan on January 12, 2008, to further collaboration.[55]

John J. DeGioia, Georgetown's first lay president, has led the school since 2001. DeGioia has continued its financial modernization and has sought to "expand opportunities for intercultural and interreligious dialogue."[56] DeGioia also founded the annual Building Bridges Seminar in 2001, which brings global religious leaders together, and is part of Georgetown's effort to promote religious pluralism.[57] The Berkley Center for Religion, Peace, and World Affairs was begun as an initiative in 2004, and after a grant from William R. Berkley, was launched as an independent organization in 2006.[57] Additionally, The Center for International and Regional Studies opened in 2005 at the new Qatar campus.[58] Between 2012 and 2018, Georgetown received more than $350 million from Gulf Cooperation Council countries including Saudi Arabia, Qatar, and the United Arab Emirates.[59]
    """
    , "who is John J DeGioia?")

print(a)
print(f"GENERATED SUMMARY in {time.time()-start_time}sec")