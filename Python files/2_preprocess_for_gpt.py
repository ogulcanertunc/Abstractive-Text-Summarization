import pandas as pd
import numpy as np
import timeit
import re

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import nltk
nltk.download('punkt')

import textblob
from textblob import TextBlob



import torch
print(torch.__version__,' pytorch version')
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

!pip install transformers==2.6.0


import transformers
print(transformers.__version__,' make sure transformers version is 2.6.0')
from transformers import GPT2Tokenizer


### 1 Necessary Functions ###

### 1.1 Requirement Mini Functions ###

def load_words(df, num, with_title=False):
    """import dataframe with number of what sample to choose,
    return a keyword (together with title or not) as strings
    and abstract for summarization.
    and 3 distractors. all as a tuple of 5 strings"""
    arr_distract = np.random.randint(len(df), size=3)
    keyword = df['keyword_POS_str'][num]
    if with_title:
        title = df['headlines'][num]
        keyword = title + keyword
    abstract = df['ctext'][num]
    part1 = df['ctext'][arr_distract[0]]
    part2 = df['ctext'][arr_distract[1]]
    part3 = df['ctext'][arr_distract[2]]

    return (keyword, abstract, part1, part2, part3)


def tag_pull_abstract(df, list_POS):
    """ return list of keyword list
    input: pandas dataframe
                    list of part of speech tag (in order to generate keyword)
    ourput: List(List(keyword string))"""
    list_tokenized = df['ctext'].apply(
        lambda x: nltk.pos_tag(nltk.word_tokenize(x))).values
    list_answer = [[item[0] for item in row if item[1] in list_POS]
                   for row in list_tokenized]
    # list_answer = list(map(lambda x: ' '.join(x), list_answer))
    return list_answer


def write_input_ids(word_batch, max_len=1024):
    """return list of input tokens"""
    keyword, abstract, dis1, dis2, dis3 = word_batch

    input_true = tokenizer.encode('<|startoftext|> ' + keyword + ' <|summarize|> ' + abstract + ' <|endoftext|>',
                                  max_length=tokenizer.max_len)
    input_dis1 = tokenizer.encode('<|startoftext|> ' + keyword + ' <|summarize|> ' + dis1 + ' <|endoftext|>',
                                  max_length=tokenizer.max_len)
    input_dis2 = tokenizer.encode('<|startoftext|> ' + keyword + ' <|summarize|> ' + dis2 + ' <|endoftext|>',
                                  max_length=tokenizer.max_len)
    input_dis3 = tokenizer.encode('<|startoftext|> ' + keyword + ' <|summarize|> ' + dis3 + ' <|endoftext|>',
                                  max_length=tokenizer.max_len)

    if max_len == None:
        max_len = max(len(input_true), len(input_dis1), len(input_dis2), len(input_dis3))
    for i in [input_true, input_dis1, input_dis2, input_dis3]:
        while len(i) < max_len:
            i.append(tokenizer.pad_token_id)
    list_input_token = [input_true, input_dis1, input_dis2, input_dis3]
    return list_input_token


def write_token_type_labels(list_input_ids, max_len=1024):
    list_segment = []
    for item in list_input_ids:
        try:
            item.index(tokenizer.eos_token_id)
        except:
            item[-1] = tokenizer.eos_token_id
        num_seg_a = item.index(tokenizer.additional_special_tokens_ids[1]) + 1
        end_index = item.index(tokenizer.eos_token_id)
        num_seg_b = end_index - num_seg_a + 1
        num_pad = max_len - end_index - 1
        segment_ids = [tokenizer.additional_special_tokens_ids[0]] * num_seg_a + [
            tokenizer.additional_special_tokens_ids[1]] * num_seg_b + [tokenizer.pad_token_id] * num_pad
        list_segment.append(segment_ids)
    return list_segment


def write_lm_labels(list_input_ids, list_type_labels):
    list_lm_label = []
    is_true_label = True
    for input_tokens, segments in zip(list_input_ids, list_type_labels):
        if is_true_label:
            is_true_label = False
            temp_list = []
            for token, segment in zip(input_tokens, segments):
                if segment == tokenizer.additional_special_tokens_ids[1]:
                    temp_list.append(token)
                else:
                    temp_list.append(-100)
            list_lm_label.append(temp_list)
        else:
            temp_list = [-100] * len(input_tokens)
            list_lm_label.append(temp_list)
    return list_lm_label


def execute_tokenization(word_tuple):
    list_input_ids = write_input_ids(word_tuple)
    list_type_labels = write_token_type_ids(list_input_ids)
    list_last_tokens = write_last_token(list_input_ids)
    list_lm_label = write_lm_labels(list_input_ids, list_type_labels)
    list_mc_labels = write_mc_labels()
    np_tuple = shuffle_batch(list_input_ids, list_type_labels,
                             list_last_tokens, list_lm_label, list_mc_labels)
    tensor_tuple = write_torch_tensor(np_tuple)
    return tensor_tuple


def write_last_token(list_input_ids):
    list_mc_token = []
    for item in list_input_ids:
        list_mc_token.append(item.index(tokenizer.eos_token_id))
    return list_mc_token


def write_mc_label():
    return [1, 0, 0, 0]


def shuffle_batch(list_input_ids, list_type_labels, list_last_tokens, list_lm_labels, list_mc_labels):
    array_input_token = np.array(list_input_ids)
    array_segment = np.array(list_type_labels)
    array_mc_token = np.array(list_last_tokens)
    array_lm_label = np.array(list_lm_labels)
    array_mc_label = np.array(list_mc_labels)

    randomize = np.arange(4)
    np.random.shuffle(randomize)

    array_input_token = array_input_token[randomize]
    array_segment = array_segment[randomize]
    array_mc_token = array_mc_token[randomize]
    array_lm_label = array_lm_label[randomize]
    array_mc_label = array_mc_label[randomize]

    return (array_input_token, array_segment, array_mc_token, array_lm_label, array_mc_label)


def write_torch_tensor(np_batch):
    torch_input_token = torch.tensor(np_batch[0], dtype=torch.long).unsqueeze(0)
    torch_segment = torch.tensor(np_batch[1], dtype=torch.long).unsqueeze(0)
    torch_mc_token = torch.tensor(np_batch[2], dtype=torch.long).unsqueeze(0)
    torch_lm_label = torch.tensor(np_batch[3], dtype=torch.long).unsqueeze(0)
    torch_mc_label = torch.tensor([np.argmax(np_batch[4])], dtype=torch.long).unsqueeze(0)
    return (torch_input_token, torch_segment, torch_mc_token, torch_lm_label, torch_mc_label)


### 2 Example Preview ###
#########################

key_batch = load_words(train_process,123)


key_batch


### 3 Masking ###
#################
def execute_all_mini_function(df):
    exist_temp_tensor = False
    exist_big_tensor = False
    start = timeit.default_timer()
    for num in range(len(df)):
        # print(num)  # I used this to find lines with errors
        word_tuple = load_words(df, num)
        if type(word_tuple[0]) != str or type(word_tuple[1]) != str:
            continue

        list_input_ids = write_input_ids(word_tuple)
        list_type_labels = write_token_type_labels(list_input_ids)
        list_lm_labels = write_lm_labels(list_input_ids, list_type_labels)
        list_last_tokens = write_last_token(list_input_ids)
        list_mc_labels = write_mc_label()

        np_tuple = shuffle_batch(list_input_ids, list_type_labels, list_last_tokens, list_lm_labels, list_mc_labels)
        tensor_tuple = write_torch_tensor(np_tuple)

        if not exist_temp_tensor:
            temp_0 = tensor_tuple[0]
            temp_1 = tensor_tuple[1]
            temp_2 = tensor_tuple[2]
            temp_3 = tensor_tuple[3]
            temp_4 = tensor_tuple[4]
            exist_temp_tensor = True
        elif exist_temp_tensor:
            temp_0 = torch.cat((temp_0, tensor_tuple[0]), 0)
            temp_1 = torch.cat((temp_1, tensor_tuple[1]), 0)
            temp_2 = torch.cat((temp_2, tensor_tuple[2]), 0)
            temp_3 = torch.cat((temp_3, tensor_tuple[3]), 0)
            temp_4 = torch.cat((temp_4, tensor_tuple[4]), 0)

        if num % 1000 == 0:
            if not exist_big_tensor:
                big_first_tensor = temp_0
                big_second_tensor = temp_1
                big_third_tensor = temp_2
                big_fourth_tensor = temp_3
                big_fifth_tensor = temp_4
                exist_temp_tensor = False
                exist_big_tensor = True
                del temp_0, temp_1, temp_2, temp_3, temp_4
            else:
                big_first_tensor = torch.cat((big_first_tensor, temp_0), 0)
                big_second_tensor = torch.cat((big_second_tensor, temp_1), 0)
                big_third_tensor = torch.cat((big_third_tensor, temp_2), 0)
                big_fourth_tensor = torch.cat((big_fourth_tensor, temp_3), 0)
                big_fifth_tensor = torch.cat((big_fifth_tensor, temp_4), 0)
                exist_temp_tensor = False
                del temp_0, temp_1, temp_2, temp_3, temp_4

            stop = timeit.default_timer()
            print('iterations ', num, ' takes ', stop - start, ' sec')
            start = timeit.default_timer()

    big_first_tensor = torch.cat((big_first_tensor, temp_0), 0)
    big_second_tensor = torch.cat((big_second_tensor, temp_1), 0)
    big_third_tensor = torch.cat((big_third_tensor, temp_2), 0)
    big_fourth_tensor = torch.cat((big_fourth_tensor, temp_3), 0)
    big_fifth_tensor = torch.cat((big_fifth_tensor, temp_4), 0)
    return big_first_tensor, big_second_tensor, big_third_tensor, big_fourth_tensor, big_fifth_tensor


tensor_1,tensor_2,tensor_3,tensor_4,tensor_5 = execute_all_mini_function(train_process)

# create a tensor dataset object
tensor_dataset = TensorDataset(tensor_1,tensor_2,tensor_3,tensor_4,tensor_5)

# save the tensor object to load later when training
torch.save(tensor_dataset, 'torch_file_from_train.pt')

tensor_1,tensor_2,tensor_3,tensor_4,tensor_5 = execute_all_mini_function(test_process)

# create a tensor dataset object
tensor_dataset = TensorDataset(tensor_1,tensor_2,tensor_3,tensor_4,tensor_5)

# save the tensor object to load later when training
torch.save(tensor_dataset, 'torch_file_from_test.pt')


### 3.1 See an example ###
##########################

item = 123
print(tensor_1[item])
print(tensor_2[item])
print(tensor_3[item])
print(tensor_4[item])
print(tensor_5[item])

print('{:>2}{:>10}{:>10}{:>10}{:>10}{:>20}{:>10}{:>20}{:>10}'.format('count','input','decoded input','input','decoded input','input','decoded input','input','decoded input'))
count = 0
for i,j,k,m in zip(tensor_1[item][1],tensor_1[item][2],tensor_2[item][2],tensor_4[item][2]):
  i = int(i)
  j = int(j)
  k = int(k)
  m = int(m)
  if i == -100:
    decode_i = 'masked'
  else:
    decode_i = tokenizer.decode(i)
  if j == -100:
    decode_j = 'masked'
  else:
    decode_j = tokenizer.decode(j)
  if k == -100:
    decode_k = 'masked'
  else:
    decode_k = tokenizer.decode(k)
  if m == -100:
    decode_m = 'masked'
  else:
    decode_m = tokenizer.decode(m)
  #print(i,j)
  print('{:>2}{:>10}{:>10}{:>10}{:>10}{:>20}{:>10}{:>20}{:>10}'.format(count,i,decode_i,j,decode_j,k,decode_k,m,decode_m))
  count += 1


















































