!pip install transformers==2.6.0

!mkdir GPT2_train
%cd GPT2_train
!GPT2_train


import numpy as np
import timeit
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import json, argparse
from transformers import get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.__version__

import transformers
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, AdamW
print('use transformers version = ',transformers.__version__) # make sure it is 2.6.0

load_model = False
load_previous_weight = False
resize_model = False


### 1 Pretrained Model setup ###
################################

model = GPT2DoubleHeadsModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}

print(len(tokenizer), 'total length of vocab') # expect 50257

special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
#special_tokens2 = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','keyword_token':'<|keyword|>','summary_token':'<|summarize|>'}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
# The newly token the last token of the vocabulary
resize_model = True

print(len(tokenizer), 'total length of vocab')
print(tokenizer.bos_token_id, 'bos_token')
print(tokenizer.eos_token_id, 'eos_token')
print(tokenizer.pad_token_id, 'pad_token')  #token for <pad>, len of all tokens in the tokenizer
print(tokenizer.additional_special_tokens_ids[0], 'keyword_token') #token for <|keyword|>
print(tokenizer.additional_special_tokens_ids[1], 'summary_token') #token for <|summarize|>


### 2. Import Torch Files ###
#############################

train_dataset = torch.load('torch_train.pt')
validation_dataset = torch.load('torch_test.pt')

train_dataset

### 3. Train a part of train dataset ###
########################################

train_dataset[2020]

### 3.1 Select a row and masked it ###
for count,i in enumerate(train_dataset[2020][3][0]):
    i = int(i)
    if i == -100:
        decode_i = 'masked'
    else:
        decode_i = tokenizer.decode(i)
    print(count,int(i), decode_i)

### 3.2 Create a Data Loader ###
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)

validation_sampler = RandomSampler(validation_dataset)
validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=1)

### 3.3 Learn the shapes of the piece we got from the dataset ###
input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = train_dataset[0]
print(input_ids.shape)
print(mc_token_ids.shape)
print(lm_labels.shape)
print(mc_labels.shape)
print(token_type_ids.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

### 3.4 Model Creation ###
model = model.to(device)
optimizer = AdamW(model.parameters(),lr=3e-5,eps=1e-8, correct_bias=True)
max_norm = 1.0
gradient_accumulation_steps = 10
total_steps = len(train_dataloader)
print('total step for learning rate scheduler = ',total_steps)

from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 50, num_training_steps = total_steps)

### 3.5 Test Run ###
test_run = train_dataset[2020]
test_run

# Forward pass
start = timeit.default_timer()
model.train()
optimizer.zero_grad()
test_run = (item.to(device) for item in test_run)
input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = test_run
input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = input_ids.to(device), mc_token_ids.to(device), lm_labels.to(device), mc_labels.to(device), token_type_ids.to(device)
outputs = model(input_ids = input_ids, mc_token_ids = mc_token_ids, mc_labels = mc_labels,lm_labels = lm_labels, token_type_ids = token_type_ids)
lm_loss, mc_loss = outputs[0], outputs[1]
#del input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels
lm_coef = 2.0
mc_coef = 1.0
total_loss = lm_loss * lm_coef + mc_loss * mc_coef
#print('lm_loss = ',lm_loss.item())
#print('mc_loss = ',mc_loss.item())
#print('total_loss = ',total_loss.item())
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
stop = timeit.default_timer()
print('1 epoch takes {:.3f}'.format(stop - start),' sec')


### 4. Full Model Training ###
##############################

!pip install pytorch-ignite

from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping

def process_function(engine,batch):
    #start = timeit.default_timer()
    model.train()
    #optimizer.zero_grad()
    batch = (item.to(device) for item in batch)
    input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
    outputs = model(input_ids = input_ids, mc_token_ids = mc_token_ids, mc_labels = mc_labels,
                  lm_labels = lm_labels, token_type_ids = token_type_ids)
    lm_loss, mc_loss = outputs[0], outputs[1]
    #del input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels
    lm_coef = 2.0
    mc_coef = 1.0
    total_loss = lm_loss * lm_coef + mc_loss * mc_coef
    total_loss = total_loss / gradient_accumulation_steps
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if engine.state.iteration % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    return lm_loss.item(),mc_loss.item(),total_loss.item()*gradient_accumulation_steps

def evaluate_function(engine,batch):
    model.eval()
    with torch.no_grad():
        batch = (item.to(device) for item in batch)
        input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
        outputs = model(input_ids = input_ids, mc_token_ids = mc_token_ids, mc_labels = mc_labels,
                  lm_labels = lm_labels, token_type_ids = token_type_ids)
        lm_loss, mc_loss = outputs[0], outputs[1]
        lm_coef = 2.0
        mc_coef = 1.0
        total_loss = lm_loss * lm_coef + mc_loss * mc_coef
    return lm_loss.item(),mc_loss.item(),total_loss.item()


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)

training_history = {'lm_loss': [], 'mc_loss': [], 'total_loss': []}
validation_history = {'lm_loss': [], 'mc_loss': [], 'total_loss': []}

RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'lm_loss')
RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'mc_loss')
RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'total_loss')


RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'lm_loss')
RunningAverage(output_transform=lambda x: x[1]).attach(evaluator, 'mc_loss')
RunningAverage(output_transform=lambda x: x[2]).attach(evaluator, 'total_loss')


@trainer.on(Events.ITERATION_COMPLETED(every=50))
def print_trainer_logs(engine):
    # try:
    #   start
    # except:
    #   start = timeit.default_timer()
    loss_LM = engine.state.metrics['lm_loss']
    loss_NSP = engine.state.metrics['mc_loss']
    combined_loss = engine.state.metrics['total_loss']
    stop = timeit.default_timer()
    print("Trainer Results - iteration {} - LM loss: {:.2f} MC loss: {:.2f} total loss: {:.2f} report time: {:.1f}"
    .format(engine.state.iteration, loss_LM, loss_NSP, combined_loss,stop))

checkpointer = ModelCheckpoint('./GPT2_train', 'GPT2_summarizer', n_saved=2, create_dir=True, save_as_state_dict=True,require_empty=False)
trainer.add_event_handler(Events.ITERATION_COMPLETED(every=15000), checkpointer, {'epoch_2': model})
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'epoch_2_done': model})

def print_logs(engine, dataloader, mode, history_dict):
    evaluator.run(dataloader, max_epochs=5)
    metrics = evaluator.state.metrics
    avg_LM_loss = metrics['lm_loss']
    avg_NSP_loss = metrics['mc_loss']
    avg_total_loss = metrics['total_loss']
    #avg_loss =  avg_bce + avg_kld
    print(
        mode + " Results - Epoch {} - Avg lm_loss: {:.2f} Avg mc_loss: {:.2f} Avg total_loss: {:.2f}"
        .format(engine.state.epoch, avg_LM_loss, avg_NSP_loss, avg_total_loss))
    for key in evaluator.state.metrics.keys():
        history_dict[key].append(evaluator.state.metrics[key])

trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, validation_dataloader, 'Validation', validation_history)

e = trainer.run(train_dataloader, max_epochs=2)


### 5. Save the Model and related files ###
###########################################
output_dir = './trained_model'

model.config.to_json_file('GPT2_train/trained_model/config.json')
tokenizer.save_vocabulary('GPT2_train/trained_model')


# save the model and tokenizer configuration
output_dir = './trained_model_dir'

model.save_pretrained(output_dir)
model.config.to_json_file('trained_model_dir/config.json')
tokenizer.save_vocabulary(output_dir)

tokenizer.save_pretrained(output_dir)
