import nltk
import streamlit as sl
nltk.download('punkt')
from Helpers.helpers import *
import torch
import textwrap
wrapper = textwrap.TextWrapper(width=70)
SEED = 1234
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
torch.manual_seed(SEED)

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
list_keywords = get_keywords(text)

GPT2_input = tokenizer_GPT2.encode('<|startoftext|> ' +title + list_keywords + ' <|summarize|> ')
GPT2_input_torch = torch.tensor(GPT2_input, dtype=torch.long)
print("the keyword input :")
wrapper.wrap(tokenizer_GPT2.decode(GPT2_input_torch))


temperature, greedy_search, top_k, top_p, max_length, min_length, num_return_sequences = get_params()

if use_GPU_GPT_generator:
    GPT2_generator = GPT2_generator.to(device)
    GPT2_input_torch = GPT2_input_torch.to(device)

do_sample = not greedy_search
if do_sample == False:
    num_return_sequences = 1


sampling_output = sampling_output_func(GPT2_generator,GPT2_input_torch, temperature, greedy_search, top_k, top_p, max_length, min_length, num_return_sequences,do_sample)

#===============================================================================================#
# Summary
#===============================================================================================#
which_output = 2
sum_text= wrapper.wrap(tokenizer_GPT2.decode(sampling_output[which_output,len(GPT2_input_torch):],
    skip_special_tokens=True)[:5000])


if sl.button('Predict'):
    sl.success(sum_text)
