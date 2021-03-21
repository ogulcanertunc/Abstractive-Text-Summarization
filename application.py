import nltk
import streamlit as sl
nltk.download('punkt')
import torch
import textwrap
wrapper = textwrap.TextWrapper(width=70)
SEED = 1234
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
torch.manual_seed(SEED)
import transformers
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from helpers.helpers import *
#===============================================================================================#

# Streamlit

#===============================================================================================#

#sl.title("Abstractive Summarizator by Ogi")

#title = sl.text_area('Enter Your Title Here')
#text = sl.text_area('Enter Your Text Here ')

title = "Father of printer"
text = "Johannes Gutenberg (1398 – 1468) was a German goldsmith and publisher who introduced printing to Europe. His introduction of mechanical movable type printing to Europe started the Printing Revolution and is widely regarded as the most important event of the modern period. It played a key role in the scientific revolution and laid the basis for the modern knowledge-based economy and the spread of learning to the masses. Gutenberg many contributions to printing are: the invention of a process for mass-producing movable type, the use of oil-based ink for printing books, adjustable molds, and the use of a wooden printing press. His truly epochal invention was the combination of these elements into a practical system that allowed the mass production of printed books and was economically viable for printers and readers alike. In Renaissance Europe, the arrival of mechanical movable type printing introduced the era of mass communication which permanently altered the structure of society. The relatively unrestricted circulation of information—including revolutionary ideas—transcended borders, and captured the masses in the Reformation. The sharp increase in literacy broke the monopoly of the literate elite on education and learning and bolstered the emerging middle class."

#===============================================================================================#

# Functions and Models Prepared

#===============================================================================================#
device = torch.device("cpu")
if use_GPU_GPT_generator:
    GPT2_generator = GPT2_generator.to(device)
    GPT2_input_torch = GPT2_input_torch.to(device)



tokenizer_GPT2, GPT2_generator,use_GPU_GPT_generator = build_model()

list_keywords = get_keywords(text)

GPT2_input = tokenizer_GPT2.encode('<|startoftext|> ' +title + list_keywords + ' <|summarize|> ')
GPT2_input_torch = torch.tensor(GPT2_input, dtype=torch.long)
print("the keyword input :")
wrapper.wrap(tokenizer_GPT2.decode(GPT2_input_torch))

temperature = 1
greedy_search = False
top_k = 35
top_p = 0.8
max_length = 200
min_length = 20
num_return_sequences = 3

do_sample = not greedy_search
if do_sample == False:
    num_return_sequences = 1


sampling_output = GPT2_generator.generate(
        input_ids=GPT2_input_torch.unsqueeze(0),
        max_length=max_length + len(GPT2_input_torch),
        min_length=min_length + len(GPT2_input_torch),
        temperature=temperature,
        decoder_start_token_id='<|summarize|>',
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=3)

#===============================================================================================#
# Summary
#===============================================================================================#
which_output = 2
sum_text= wrapper.wrap(tokenizer_GPT2.decode(sampling_output[which_output,len(GPT2_input_torch):],
    skip_special_tokens=True)[:5000])


if sl.button('Predict'):
    sl.success(sum_text)
