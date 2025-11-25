import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset,load_from_disk
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
'''
4．使用LoRA微调大型语言模型以执行文本摘要任务。假设你已经加载了预训练的BART模型，现在想通过LoRA微调使其适应文本摘要任务。
原文本：“自然语言处理（Natural Language Processing，NLP）是AI领域中与计算机和人类语言之间交互的研究。NLP的目标是使计算机能够理解、解释、生成人类语言，使计算机与人的交互更加自然。它涉及文本处理、语音处理、机器翻译等多个方面的任务。”使用LoRA微调方法，将上述文本进行摘要生成，提取出主要信息。
'''
# 加载T5模型和Tokenizer
# model_name = "t5-small"  # 可以替换为更大的模型，比如t5-base或t5-large
# model_name = "D:/data/PEFT/t5-base"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# 加载预训练的BART模型和tokenizer
model_name = "D:/data/PEFT/facebook/bart-large-cnn"
# 使用 facebook/bart-large-cnn 模型，它是专门为文本摘要任务优化的 BART 模型。BartTokenizer 用于将文本转换为模型能够理解的 tokens。
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
# 设置Lora配置
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=16,  # 权重衰减系数
    lora_dropout=0.1,  # dropout率
    bias="none",  # 是否使用偏置
    task_type="SEQ_2_SEQ_LM"  # 任务类型：序列到序列
)

# 使用Lora进行微调
peft_model = get_peft_model(model, lora_config)
''' '''
# 加载数据集，这里使用是 "cnn_dailymail" 数据集作为摘要任务
# dataset = load_dataset("cnn_dailymail", "3.0.0")
# D:/data/PEFT/cnn_dailymail/3.0.0
#TODO：加载本地数据集，注意是 parquet 文件
# dataset = load_dataset("D:/data/PEFT/cnn_dailymail","3.0.0")
# arrow_dir='D:/data/PEFT/cnn_dailymail/3.0.0/cache/arrow/cnn_dailymail-002d97601cfece2d/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137/'
arrow_dir='D:/data/PEFT/cnn_dailymail/3.0.0/arrow/'
# dataset = load_dataset("D:/data/PEFT/cnn_dailymail","3.0.0",cache_dir='D:/data/PEFT/cnn_dailymail/3.0.0/cache')
# dataset.save_to_disk(arrow_dir)
# print('save_to_disk ---------- ')
#TODO:经过load_dataset：Dataset arrow downloaded and prepared to
# D:/data/PEFT/cnn_dailymail/3.0.0/cache/arrow/cnn_dailymail-002d97601cfece2d/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137.
# Subsequent calls will reuse this data.
dataset=load_from_disk(arrow_dir)
#type(dataset):<class 'datasets.dataset_dict.DatasetDict'>
# dataset.keys():dict_keys(['train', 'test'])
# 如果是单个数据集（而不是数据集字典），可以这样加载：
# dataset = Dataset.load_from_disk(arrow_dir)
# 如果是多个数据集（例如训练集、验证集等），可以这样加载：
# dataset_Dict = DatasetDict.load_from_disk(arrow_dir)

# 数据处理函数
def preprocess_function(examples):#examples{'article':[list:1000],'highlights':[list:1000],'id'=[list:1000]}
    inputs = [doc for doc in examples["article"]]# inputs={list:1000}
    targets = [summarize for summarize in examples["highlights"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")#{dict:2}={'input_idx':[],'attention_mask':[]}
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")
    # {dict:2}={'input_idx':[],'attention_mask':[]}
    model_inputs["labels"] = labels["input_ids"]#{dict:3}={'input_idx':[list:1000],'attention_mask':[list:1000],'labels':[list:1000]}
    return model_inputs

# 数据预处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)
# type(tokenized_datasets):<class 'datasets.dataset_dict.DatasetDict'>
# tokenized_datasets.keys() dict_keys(['train', 'test'])
# 创建DataLoader
train_dataset = tokenized_datasets["train"]# len(train_dataset)=221610
bz=16 # 8
train_dataloader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
#TODO：保存数据
def save_train_dataloader(train_loader):
    import pickle
    with open('train_loader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
def load_train_dataloader():
    import pickle
    with open("train_loader.pkl", 'rb') as f:
        train_loader = pickle.loads(f.read())
        return train_loader
# 设置优化器
optimizer = AdamW(peft_model.parameters(), lr=1e-5)

# 训练模型
peft_model.train()
epochs = 3  # 训练3个周期

for epoch in range(epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
    # for step,batch in enumerate(tqdm(train_dataloader)):
        # input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # labels = batch["labels"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        input_ids = torch.vstack(batch["input_ids"]).reshape(bz,-1) # len(batch["input_ids"])=512
        # [tensor([  41,   41,   41,   41, 3144, 1051, 2386,    3]),
        # tensor([  254,   254,   254,   254,  4975,  5901,    41, 26342]),...
        # tensor([ 5727,  2383,   592,     9, 17048,  1350,     6,    88...
        attention_mask = torch.vstack(batch["attention_mask"]).reshape(bz,-1)
        labels = torch.vstack(batch["labels"]).reshape(bz,-1)
        # input_ids = input_ids.permute(1, 0)
        # attention_mask=attention_mask.permute(1, 0)
        # labels = labels.permute(1, 0)
        # print('input_ids.shape:', input_ids.shape,',attention_mask.shape:',attention_mask.shape,',labels.shape:',labels.shape)
        optimizer.zero_grad()
        # 前向传播
        # input_ids.shape: torch.Size([8, 512]) ,attention_mask.shape: torch.Size([8, 512]) ,labels.shape: torch.Size([8, 150])
        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # 反向传播
        loss.backward()
        optimizer.step()
        print('loss=',loss)
        progress_bar.set_postfix(loss=loss.item())

# 评估模型
peft_model.eval()

# 示例输入进行摘要生成
text = "The quick brown fox jumps over the lazy dog. The dog barked at the fox, but the fox ran away."

inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
# input_ids = inputs["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# attention_mask = inputs["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# input_ids = torch.vstack(inputs["input_ids"]).reshape(bz,-1)
# attention_mask = torch.vstack(inputs["attention_mask"]).reshape(bz,-1)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
# 使用T5生成摘要
summary_ids = peft_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=4, max_length=150,
                                  length_penalty=2.0, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Generated Summary:", summary)

''' 2024-12-03
# Epochs=0 不微调
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/exercise/pythonExercise/paperWork/PEFT_Class/ch7_BART_T5_LoRA_Summary.py
Generated Summary: The quick brown fox jumps over the lazy dog. The dog barked at the fox, but the fox ran away. The fox was caught on camera running away from the dog. It was captured on camera by the dog's owner, who posted it on Facebook. The video has been viewed more than 100,000 times.

Process finished with exit code 0
'''


'''
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/data/PEFT/My_ChatGLM_6B_Lora_Tuning_En_And_Zh-main/ch7_BART_T5_LoRA_Summary.py
Loading cached processed dataset at D:\data\PEFT\cnn_dailymail\3.0.0\arrow\train\cache-949776892ca4008e.arrow
Loading cached processed dataset at D:\data\PEFT\cnn_dailymail\3.0.0\arrow\test\cache-49640f7deed8fa62.arrow
Generated Summary: The dog barked at the fox, but the fox ran away.

Process finished with exit code 0
'''

'''11-13 14:57
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/data/PEFT/My_ChatGLM_6B_Lora_Tuning_En_And_Zh-main/ch7_BART_T5_LoRA_Summary.py
load_from_disk:  dict_keys(['train', 'test'])
Loading cached processed dataset at D:\data\PEFT\cnn_dailymail\3.0.0\arrow\train\cache-949776892ca4008e.arrow
Loading cached processed dataset at D:\data\PEFT\cnn_dailymail\3.0.0\arrow\test\cache-49640f7deed8fa62.arrow
after preprocess_function, get tokenized_datasets,  <class 'datasets.dataset_dict.DatasetDict'>
get train_dataloader
Epoch 1:   0%|          | 0/27702 [00:00<?, ?it/s]
┌───────────────────── Traceback (most recent call last) ─────────────────────┐
│ D:/data/PEFT/My_ChatGLM_6B_Lora_Tuning_En_And_Zh-main/ch7_BART_T5_LoRA_Summary.py:7 │
│ 9 in <module>                                                               │
│                                                                             │
│    76 │   │   # input_ids = batch["input_ids"].to(torch.device("cuda" if to │
│    77 │   │   # attention_mask = batch["attention_mask"].to(torch.device("c │
│    78 │   │   # labels = batch["labels"].to(torch.device("cuda" if torch.cu │
│ >  79 │   │   input_ids = torch.tensor(batch["input_ids"],dtype=torch.long) │
│    80 │   │   attention_mask = torch.tensor(batch["attention_mask"],dtype=t │
│    81 │   │   labels = torch.tensor(batch["labels"],dtype=torch.long)       │
│    82                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
ValueError: only one element tensors can be converted to Python scalars

Process finished with exit code 1

'''

'''11-13  13:30
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/data/PEFT/My_ChatGLM_6B_Lora_Tuning_En_And_Zh-main/ch7_BART_T5_LoRA_Summary.py
Resolving data files: 100%|██████████| 35/35 [00:00<?, ?it/s]
Downloading and preparing dataset arrow/cnn_dailymail to D:/data/PEFT/cnn_dailymail/3.0.0/cache/arrow/cnn_dailymail-c7da46093d1f7df4/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137...
Downloading data files: 100%|██████████| 3/3 [00:00<00:00, 1528.72it/s]
Extracting data files: 100%|██████████| 3/3 [00:00<00:00,  5.68it/s]
  0%|          | 0/3 [00:00<?, ?it/s]Dataset arrow downloaded and prepared to D:/data/PEFT/cnn_dailymail/3.0.0/cache/arrow/cnn_dailymail-c7da46093d1f7df4/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137. Subsequent calls will reuse this data.
100%|██████████| 3/3 [00:12<00:00,  4.28s/it]
Epoch 1:   0%|          | 0/387818 [00:00<?, ?it/s]get train_dataloader
Epoch 1:   0%|          | 0/387818 [00:00<?, ?it/s]
┌───────────────────── Traceback (most recent call last) ─────────────────────┐
│ D:/data/PEFT/My_ChatGLM_6B_Lora_Tuning_En_And_Zh-main/ch7_BART_T5_LoRA_Summary.py:8 │
│ 7 in <module>                                                               │
│                                                                             │
│    84 │   │   optimizer.zero_grad()                                         │
│    85 │   │                                                                 │
│    86 │   │   # 前向传播                                                    │
│ >  87 │   │   outputs = peft_model(input_ids=input_ids, attention_mask=atte │
│    88 │   │   loss = outputs.loss                                           │
│    89 │   │                                                                 │
│    90 │   │   # 反向传播                                                    │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\torch\nn\mod │
│ ules\module.py:1194 in _call_impl                                           │
│                                                                             │
│   1191 │   │   # this function, and just call forward.                      │
│   1192 │   │   if not (self._backward_hooks or self._forward_hooks or self. │
│   1193 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hook │
│ > 1194 │   │   │   return forward_call(*input, **kwargs)                    │
│   1195 │   │   # Do not call functions when jit is used                     │
│   1196 │   │   full_backward_hooks, non_full_backward_hooks = [], []        │
│   1197 │   │   if self._backward_hooks or _global_backward_hooks:           │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\peft\peft_mo │
│ del.py:880 in forward                                                       │
│                                                                             │
│    877 │   │   │   │   output_attentions=output_attentions,                 │
│    878 │   │   │   │   output_hidden_states=output_hidden_states,           │
│    879 │   │   │   │   return_dict=return_dict,                             │
│ >  880 │   │   │   │   **kwargs,                                            │
│    881 │   │   │   )                                                        │
│    882 │   │                                                                │
│    883 │   │   batch_size = input_ids.shape[0]                              │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\torch\nn\mod │
│ ules\module.py:1194 in _call_impl                                           │
│                                                                             │
│   1191 │   │   # this function, and just call forward.                      │
│   1192 │   │   if not (self._backward_hooks or self._forward_hooks or self. │
│   1193 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hook │
│ > 1194 │   │   │   return forward_call(*input, **kwargs)                    │
│   1195 │   │   # Do not call functions when jit is used                     │
│   1196 │   │   full_backward_hooks, non_full_backward_hooks = [], []        │
│   1197 │   │   if self._backward_hooks or _global_backward_hooks:           │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\transformers │
│ \models\t5\modeling_t5.py:1690 in forward                                   │
│                                                                             │
│   1687 │   │   │   │   head_mask=head_mask,                                 │
│   1688 │   │   │   │   output_attentions=output_attentions,                 │
│   1689 │   │   │   │   output_hidden_states=output_hidden_states,           │
│ > 1690 │   │   │   │   return_dict=return_dict,                             │
│   1691 │   │   │   )                                                        │
│   1692 │   │   elif return_dict and not isinstance(encoder_outputs, BaseMod │
│   1693 │   │   │   encoder_outputs = BaseModelOutput(                       │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\torch\nn\mod │
│ ules\module.py:1194 in _call_impl                                           │
│                                                                             │
│   1191 │   │   # this function, and just call forward.                      │
│   1192 │   │   if not (self._backward_hooks or self._forward_hooks or self. │
│   1193 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hook │
│ > 1194 │   │   │   return forward_call(*input, **kwargs)                    │
│   1195 │   │   # Do not call functions when jit is used                     │
│   1196 │   │   full_backward_hooks, non_full_backward_hooks = [], []        │
│   1197 │   │   if self._backward_hooks or _global_backward_hooks:           │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\transformers │
│ \models\t5\modeling_t5.py:977 in forward                                    │
│                                                                             │
│    974 │   │   │   │   f"You cannot specify both {err_msg_prefix}input_ids  │
│    975 │   │   │   )                                                        │
│    976 │   │   elif input_ids is not None:                                  │
│ >  977 │   │   │   input_shape = input_ids.size()                           │
│    978 │   │   │   input_ids = input_ids.view(-1, input_shape[-1])          │
│    979 │   │   elif inputs_embeds is not None:                              │
│    980 │   │   │   input_shape = inputs_embeds.size()[:-1]                  │
└─────────────────────────────────────────────────────────────────────────────┘
AttributeError: 'list' object has no attribute 'size'

Process finished with exit code 1


'''

''' 11-12
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/data/PEFT/My_ChatGLM_6B_Lora_Tuning_En_And_Zh-main/ch7_BART_T5_LoRA_Summary.py
Resolving data files: 100%|██████████| 26/26 [00:00<?, ?it/s]
Downloading and preparing dataset arrow/cnn_dailymail to D:/data/PEFT/cnn_dailymail/3.0.0/cache/arrow/cnn_dailymail-002d97601cfece2d/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137...
Downloading data files: 100%|██████████| 3/3 [00:00<00:00, 1504.23it/s]
Extracting data files: 100%|██████████| 3/3 [00:00<00:00, 55.97it/s]
  0%|          | 0/3 [00:00<?, ?it/s]Dataset arrow downloaded and prepared to D:/data/PEFT/cnn_dailymail/3.0.0/cache/arrow/cnn_dailymail-002d97601cfece2d/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137. Subsequent calls will reuse this data.
100%|██████████| 3/3 [00:09<00:00,  3.02s/it]
Epoch 1:   0%|          | 0/287113 [00:00<?, ?it/s]
┌───────────────────── Traceback (most recent call last) ─────────────────────┐
│ D:/data/PEFT/My_ChatGLM_6B_Lora_Tuning_En_And_Zh-main/ch7_BART_T5_LoRA_Summary.py:7 │
│ 3 in <module>                                                               │
│                                                                             │
│    70 │   │   optimizer.zero_grad()                                         │
│    71 │   │                                                                 │
│    72 │   │   # 前向传播                                                    │
│ >  73 │   │   outputs = peft_model(input_ids=input_ids, attention_mask=atte │
│    74 │   │   loss = outputs.loss                                           │
│    75 │   │                                                                 │
│    76 │   │   # 反向传播                                                    │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\torch\nn\mod │
│ ules\module.py:1194 in _call_impl                                           │
│                                                                             │
│   1191 │   │   # this function, and just call forward.                      │
│   1192 │   │   if not (self._backward_hooks or self._forward_hooks or self. │
│   1193 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hook │
│ > 1194 │   │   │   return forward_call(*input, **kwargs)                    │
│   1195 │   │   # Do not call functions when jit is used                     │
│   1196 │   │   full_backward_hooks, non_full_backward_hooks = [], []        │
│   1197 │   │   if self._backward_hooks or _global_backward_hooks:           │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\peft\peft_mo │
│ del.py:880 in forward                                                       │
│                                                                             │
│    877 │   │   │   │   output_attentions=output_attentions,                 │
│    878 │   │   │   │   output_hidden_states=output_hidden_states,           │
│    879 │   │   │   │   return_dict=return_dict,                             │
│ >  880 │   │   │   │   **kwargs,                                            │
│    881 │   │   │   )                                                        │
│    882 │   │                                                                │
│    883 │   │   batch_size = input_ids.shape[0]                              │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\torch\nn\mod │
│ ules\module.py:1194 in _call_impl                                           │
│                                                                             │
│   1191 │   │   # this function, and just call forward.                      │
│   1192 │   │   if not (self._backward_hooks or self._forward_hooks or self. │
│   1193 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hook │
│ > 1194 │   │   │   return forward_call(*input, **kwargs)                    │
│   1195 │   │   # Do not call functions when jit is used                     │
│   1196 │   │   full_backward_hooks, non_full_backward_hooks = [], []        │
│   1197 │   │   if self._backward_hooks or _global_backward_hooks:           │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\transformers │
│ \models\t5\modeling_t5.py:1690 in forward                                   │
│                                                                             │
│   1687 │   │   │   │   head_mask=head_mask,                                 │
│   1688 │   │   │   │   output_attentions=output_attentions,                 │
│   1689 │   │   │   │   output_hidden_states=output_hidden_states,           │
│ > 1690 │   │   │   │   return_dict=return_dict,                             │
│   1691 │   │   │   )                                                        │
│   1692 │   │   elif return_dict and not isinstance(encoder_outputs, BaseMod │
│   1693 │   │   │   encoder_outputs = BaseModelOutput(                       │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\torch\nn\mod │
│ ules\module.py:1194 in _call_impl                                           │
│                                                                             │
│   1191 │   │   # this function, and just call forward.                      │
│   1192 │   │   if not (self._backward_hooks or self._forward_hooks or self. │
│   1193 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hook │
│ > 1194 │   │   │   return forward_call(*input, **kwargs)                    │
│   1195 │   │   # Do not call functions when jit is used                     │
│   1196 │   │   full_backward_hooks, non_full_backward_hooks = [], []        │
│   1197 │   │   if self._backward_hooks or _global_backward_hooks:           │
│                                                                             │
│ D:\software\python\pycharmVenv\python37tf114\lib\site-packages\transformers │
│ \models\t5\modeling_t5.py:977 in forward                                    │
│                                                                             │
│    974 │   │   │   │   f"You cannot specify both {err_msg_prefix}input_ids  │
│    975 │   │   │   )                                                        │
│    976 │   │   elif input_ids is not None:                                  │
│ >  977 │   │   │   input_shape = input_ids.size()                           │
│    978 │   │   │   input_ids = input_ids.view(-1, input_shape[-1])          │
│    979 │   │   elif inputs_embeds is not None:                              │
│    980 │   │   │   input_shape = inputs_embeds.size()[:-1]                  │
└─────────────────────────────────────────────────────────────────────────────┘
AttributeError: 'list' object has no attribute 'size'

Process finished with exit code 1

'''



