import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PrefixTuningConfig, get_peft_model
from datasets import load_dataset,load_from_disk
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
'''
3．使用前缀微调方法微调大型语言模型以执行中文翻译成英文的任务。
加载了预训练的T5模型，+前缀微调方法使其适应zh-en的任务。
原文本：“这是一个关于大模型微调的问题，希望通过前缀微调方法来适应中文翻译成英文的任务。”
代码分析参考：https://www.cnblogs.com/tuyuge/p/17612914.html
'''
'''T5（Text-to-Text Transfer Transformer）属于 encoder-decoder 架构。
Encoder（编码器）：负责处理输入序列，将其转换为上下文感知的表示（即隐藏状态）。
Decoder（解码器）：在获取输入的表示后，生成目标序列的输出。
备注：
（1）自回归模型：像 GPT 这样的模型通常是自回归的，即每一步生成的输出会作为下一个时间步的输入。这些模型通常只使用解码器部分。
针对自回归模型，在句子前面添加前缀，得到 z = [PREFIX; x; y]，合适的上文能够在固定 LM 的情况下去引导生成下文（比如：GPT3的上下文学习）
（2）Encoder-Decoder 模型：T5 模型同时使用编码器和解码器，它依赖于编码器处理输入信息，并利用解码器来生成输出。
针对编码器-解码器架构：Encoder和Decoder都增加了前缀，得到 z = [PREFIX; x; PREFIX0; y]。
Encoder端增加前缀是为了引导输入部分的编码，Decoder 端增加前缀是为了引导后续token的生成。
'''
'''BART（Bidirectional and Auto-Regressive Transformers）属于 encoder-decoder 架构
BART 结合了 自回归 和 双向 的特性，但在架构上，它仍然是一个 encoder-decoder 模型。
Encoder（编码器）：BART 的编码器是一个类似于 BERT 的双向 Transformer 编码器，对输入文本的双向表示进行编码，捕获上下文信息。
Decoder（解码器）：BART 的解码器是自回归的，即在生成每个新的词时，它会依赖之前生成的词，这类似于 GPT 的解码器结构。
'''
# 1. 加载T5模型和Tokenizer
# model_name = "t5-small"  # 你可以选择其他版本，如 t5-base、t5-large 等
model_name = "D:/data/PEFT/t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# peft_config = PrefixTuningConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False, num_virtual_tokens=8)
'''AutoModelForSeq2SeqLM 是一个通用模型接口，旨在加载任何支持 Seq2Seq 任务的模型架构。
它不局限于某一特定模型，而是支持多个类型的模型，例如：
    BART（适合文本生成、摘要、机器翻译等）
    T5（适合文本到文本转换任务，如翻译、摘要、问答等）
    MarianMT（用于机器翻译）
    BART、PEGASUS 等
这个接口可以自动根据模型名称选择对应架构的实现，方便用户切换不同的 Seq2Seq 模型。
T5ForConditionalGeneration:专门针对 T5 模型设计的类，用于生成条件文本输出，具备专门的优化和预训练权重。
T5 将所有任务（如翻译、摘要、文本生成等）视为文本到文本的转换任务。
T5ForConditionalGeneration 主要用于 条件生成 任务，输入是一个序列，模型根据输入序列生成一个相应的输出序列。
'''
# # 1. 加载 mBART 模型和 tokenizer
# from transformers import MBartForConditionalGeneration, MBartTokenizer
# model_name = "facebook/mbart-large-50-many-to-many-mmt"  # 支持多语言的 mBART 模型
# tokenizer = MBartTokenizer.from_pretrained(model_name)
# model = MBartForConditionalGeneration.from_pretrained(model_name)
#TODO：1、设置 PrefixTuning 配置
prefix_tuning_config = PrefixTuningConfig(
    task_type="SEQ_2_SEQ_LM",  # 指定任务类型，序列到序列任务
    num_layers=4,  # 选择微调模型的层数
    num_virtual_tokens=8, # 虚拟token的数量，换句话说就是提示（prompt）
    prefix_projection=True,#是否投影前缀嵌入(token)，默认值为False，表示使用P-Tuning v2， 如果为true，则表示使用 Prefix Tuning。
)# inference_mode：是否在推理模式下使用Peft模型。
'''
(prompt_encoder): ModuleDict(
    (default): PrefixEncoder(
      (embedding): Embedding(8, 6144)
    )
  )
  (word_embeddings): Embedding(32128, 768)
trainable params: 49152 || all params: 222952704 || trainable%: 0.022045931320034583
prefix_tuning_config = PrefixTuningConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False, num_virtual_tokens=20)
(prompt_encoder): ModuleDict(
    (default): PrefixEncoder(
      (embedding): Embedding(20, 18432)
    )
  )
  (word_embeddings): Embedding(32128, 768)
trainable params: 368640 || all params: 223272192 || trainable%: 0.1651078876853594
prefix_tuning_config = PrefixTuningConfig(
    task_type="SEQ_2_SEQ_LM",  # 指定任务类型，序列到序列任务
    num_layers=4,  # 选择微调模型的层数
    num_virtual_tokens=8, # 虚拟token的数量，换句话说就是提示（prompt）
    prefix_projection=True,
)
trainable params: 5321472 || all params: 228225024 || trainable%: 2.3316777041942607
'''
#TODO：2、使用 PrefixTuning 微调模型
peft_model = get_peft_model(model, prefix_tuning_config)
# print('peft_model:',peft_model)
peft_model.print_trainable_parameters()
# Prefix Tuning 可训练参数的数量(仅为368640)以及占比（仅为0.1651%）
# 加载数据集：以中英文翻译为例，我们使用英语和中文的翻译数据集
# dataset = load_dataset("wmt16", "de-en")  # 你可以选择其他翻译数据集，中文-英文可使用其他数据集
#TODO: 下载路径：https://huggingface.co/datasets/wmt/wmt16/tree/main
# https://huggingface.co/datasets/wmt/wmt16/tree/main/de-en
# 加载本地数据集，注意是 parquet 文件
# default:"C:/Users/86157/.cache/huggingface/datasets"---->D:/data/PEFT/wmt16/de-en/cache
# de-en: 德语->英文
# 机器翻译数据集WMT预处理流程：https://blog.csdn.net/fuhanghang/article/details/142361424
# dataset = load_dataset("D:/data/PEFT/wmt16/","de-en",cache_dir='D:/data/PEFT/wmt16/de-en/cache')
# arrow_dir='D:/data/PEFT/wmt16/de-en/arrow/'
# dataset.save_to_disk(arrow_dir)
# TODO:console:DatasetGenerationError: An error occurred while generating the dataset
# dataset=load_from_disk(arrow_dir)
# dataset = load_dataset("D:/data/PEFT/wmt16","ro-en",cache_dir='D:/data/PEFT/wmt16/ro-en/cache')
#TODO: 3. 加载数据集
# dataset = load_dataset('D:/data/PEFT/en-zh-dataset',cache_dir='D:/data/PEFT/en-zh-dataset/cache')
arrow_dir='D:/data/PEFT/en-zh-dataset/arrow/'
# dataset.save_to_disk(arrow_dir)
dataset=load_from_disk(arrow_dir)

def preprocess_function(examples):
    # arrow_dir='D:/data/PEFT/en-zh-dataset/arrow/'
    inputs = [doc['zh'] for doc in examples["translation"]]  # 中文输入
    targets = [summarize['en'] for summarize in examples["translation"]]  # 英文输出
    # inputs = [doc['en'] for doc in examples["translation"]]  # 英文输入
    # targets = [summarize['de'] for summarize in examples["translation"]]  # 德文输出
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
'''以英语（en）德语（de）翻译为例:
{'translation': [{'de': 'Wiederaufnahme der Sitzungsperiode', 'en': 'Resumption of the session'}, {'de': 'Ich erkläre die..., daß Sie schöne Ferien hatten.', 'en': 'I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.'}, ...}
'''
#TODO:4、数据预处理, examples["translation"]
tokenized_datasets = dataset.map(preprocess_function, batched=True)
# 创建DataLoader
train_dataset = tokenized_datasets["train"]
bz=8
train_dataloader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
# print(train_dataloader.shape())
#TODO：5、设置优化器
optimizer = AdamW(peft_model.parameters(), lr=1e-5)

#TODO：6、训练模型
peft_model.train()
epochs = 1  # 训练3个周期

for epoch in range(epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for batch in progress_bar:
        # input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # labels = batch["labels"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        input_ids = torch.vstack(batch["input_ids"]).reshape(bz, -1)
        attention_mask = torch.vstack(batch["attention_mask"]).reshape(bz, -1)
        labels = torch.vstack(batch["labels"]).reshape(bz, -1)
        optimizer.zero_grad()
        # 前向传播
        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

#TODO：7、评估模型
peft_model.eval()

# 示例输入进行翻译
# text = "Hello, how are you?"
text = "你好，你叫什么？"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
# input_ids = inputs["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# attention_mask = inputs["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
# 使用T5生成翻译结果
output_ids = peft_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=4, max_length=512,
                                 length_penalty=2.0, early_stopping=True)
#TODO：8、对生成的目标语言进行解码工作，就可得到目标语言的文本，并打印
# translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
translated_text = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)
print(f"Translated text: {translated_text}")
#TODO：参考：https://blog.csdn.net/2301_78285120/article/details/132701916
# 9、模型训练完成，保存高效微调部分的模型权重以供模型推理
# 注意：这里只会保存经过训练的增量 PEFT 权重。其中，adapter_config.json 为 P-Tuning v2 / Prefix Tuning 配置文件；adapter_model.bin 为 P-Tuning v2 / Prefix Tuning 权重文件。
# #peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
peft_model_id = "D:/data/PEFT/t5-base/T5_base_Prefix_SEQ_2_SEQ_LM_Tans/"
model.save_pretrained(peft_model_id)
'''
D:/data/PEFT/t5-base/T5_base_Prefix_SEQ_2_SEQ_LM_Tans/
├── [ 390]  adapter_config.json
├── [5.6M]  adapter_model.bin
└── [  93]  README.md

0 directories, 3 files
'''
def load_model_inference():
    from peft import PeftModel, PeftConfig
    # peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
    peft_model_id = "D:/data/PEFT/t5-base/T5_base_PrefixTuningConfig_SEQ_2_SEQ_LM/"
    config = PeftConfig.from_pretrained(peft_model_id)
    # 加载基础模型
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # 加载PEFT模型
    model = PeftModel.from_pretrained(model, peft_model_id)
    # 编码
    # inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
    inputs = tokenizer(f'1: first Label', return_tensors="pt")
    # 模型推理
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10,
        eos_token_id=3
    )
    # 解码
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

'''
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/exercise/pythonExercise/paperWork/PEFT_Class/ch7_T5_Prefix_Translate.py
trainable params: 5321472 || all params: 228225024 || trainable%: 2.3316777041942607
Loading cached processed dataset at D:\data\PEFT\en-zh-dataset\arrow\train\cache-f5f238415d6675c5.arrow
Epoch 1:   1%|          | 94/12068 [43:17<92:52:23, 27.92s/it, loss=1.95]
'''

'''11-13  20:30
trainable params: 49152 || all params: 222952704 || trainable%: 0.022045931320034583
Translated text: ['<pad>', 'senzati', 'senzati', 'senzati',... 'senzati','senzati', 'senzati']
'''

'''11-13  20:15 设置 epochs = 0
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/exercise/pythonExercise/paperWork/ch7_T5_Prefix_Translate.py
trainable params: 49152 || all params: 222952704 || trainable%: 0.022045931320034583
Loading cached processed dataset at D:\data\PEFT\en-zh-dataset\arrow\train\cache-2b712b8a9866d3bf.arrow
Translated text: senzati senzati senzati ... senzati senzati senzati senzati

Process finished with exit code 0

'''

'''11-13  19:49
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/exercise/pythonExercise/paperWork/ch7_T5_Prefix_Translate.py
trainable params: 49152 || all params: 222952704 || trainable%: 0.022045931320034583
Loading cached processed dataset at D:\data\PEFT\en-zh-dataset\arrow\train\cache-2b712b8a9866d3bf.arrow
Epoch 1:   0%|          | 4/12068 [01:36<80:31:19, 24.03s/it, loss=2.83]
...
Epoch 1:   0%|          | 29/12068 [13:41<96:09:58, 28.76s/it, loss=2.4]

'''

''' 11-13  19:30
D:\software\python\pycharmVenv\python37tf114\Scripts\python.exe D:/exercise/pythonExercise/paperWork/ch7_T5_Prefix_Translate.py
Map:   0%|          | 0/96538 [00:00<?, ? examples/s]trainable params: 49152 || all params: 222952704 || trainable%: 0.022045931320034583
load_from_disk
Epoch 1:   0%|          | 0/12068 [00:00<?, ?it/s]
┌───────────────────── Traceback (most recent call last) ─────────────────────┐
│ D:/exercise/pythonExercise/paperWork/ch7_T5_Prefix_Translate.py:131 in         │
│ <module>                                                                    │
│                                                                             │
│   128 │   progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")  │
│   129 │                                                                     │
│   130 │   for batch in progress_bar:                                        │
│ > 131 │   │   input_ids = batch["input_ids"].to(torch.device("cuda" if torc │
│   132 │   │   attention_mask = batch["attention_mask"].to(torch.device("cud │
│   133 │   │   labels = batch["labels"].to(torch.device("cuda" if torch.cuda │
│   134                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
AttributeError: 'list' object has no attribute 'to'

Process finished with exit code 1

'''

