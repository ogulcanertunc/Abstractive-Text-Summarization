import transformers
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
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

def build_model():
    GPT2_config_directory = 'GPT2_dir'
    tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(GPT2_config_directory)
    special_tokens = {'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'pad_token': '<pad>',
                      'additional_special_tokens': ['<|keyword|>', '<|summarize|>']}
    tokenizer_GPT2.add_special_tokens(special_tokens)
    use_GPU_GPT_generator = False
    GPT2_generator = GPT2DoubleHeadsModel.from_pretrained("Conf_gen", from_tf = False)
    return tokenizer_GPT2, GPT2_generator, use_GPU_GPT_generator
