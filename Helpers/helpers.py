import transformers
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, DistilBertModel, DistilBertTokenizer, BertTokenizer, BertForTokenClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def get_keywords(text_):
    some_text = text_
    lowered = some_text.lower()
    tokens = nltk.tokenize.word_tokenize(lowered)
    keywords = [keyword for keyword in tokens if keyword.isalpha() and not keyword in stop_words]
    keywords_string = ','.join(keywords)
    return keywords_string

def tokenizer_ogi():
    GPT2_config_directory = 'GPT2_dir'
    print('----loading GPT2 summary generator----')
    tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(GPT2_config_directory)
    special_tokens = {'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'pad_token': '<pad>',
                      'additional_special_tokens': ['<|keyword|>', '<|summarize|>']}
    tokenizer_GPT2.add_special_tokens(special_tokens)
    GPT2_generator = GPT2DoubleHeadsModel.from_pretrained(GPT2_config_directory)
    print('----GPT2 summary generator directory is true----')
    return tokenizer_GPT2,special_tokens, GPT2_generator

def sampling_output_func(GPT2_generator,GPT2_input_torch,temperature, greedy_search, top_k, top_p, max_length, min_length, num_return_sequences,do_sample):
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
    return sampling_output

def get_params():
    temperature = 1
    greedy_search = False
    top_k = 35
    top_p = 0.8
    max_length = 200
    min_length = 20  # @param {type:"integer",max:1}
    num_return_sequences = 3  # @param {type:"integer",min:1}
    return temperature, greedy_search, top_k, top_p, max_length, min_length, num_return_sequences