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

#===============================================================================================#

# Streamlit

#===============================================================================================#

sl.title("Abstractive Summarizator by Ogi")

title = sl.text_area('Enter Your Title Here')
text = sl.text_area('Enter Your Text Here ')
#===============================================================================================#

# Functions and Models Prepared

#===============================================================================================#

GPT2_config_directory = 'GPT2_dir'
tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(GPT2_config_directory)
special_tokens = {'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'pad_token': '<pad>',
                  'additional_special_tokens': ['<|keyword|>', '<|summarize|>']}
tokenizer_GPT2.add_special_tokens(special_tokens)
GPT2_generator = GPT2DoubleHeadsModel.from_pretrained(GPT2_config_directory)
use_GPU_GPT_generator = False
def get_keywords(text_):
    some_text = text_
    lowered = some_text.lower()
    tokens = nltk.tokenize.word_tokenize(lowered)
    keywords = [keyword for keyword in tokens if keyword.isalpha() and not keyword in stop_words]
    keywords_string = ','.join(keywords)
    return keywords_string
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


if use_GPU_GPT_generator:
    GPT2_generator = GPT2_generator.to(device)
    GPT2_input_torch = GPT2_input_torch.to(device)

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
