''' 16个学时，16个实践课
实用的内容和考核
ChatGPT如何使用（PPT）？具身智能，3D AIGC，韩晓光视频（实践课放视频），ch4，怎么用，现在的大模型使用，ch7，ch9大模型应用
多个课题选（分是否计算机专业）：考核怎么利用GPT写代码或者写专业资料
网上找找：如何使用GPT的PPT？
考核：技术报告（论文格式），大模型领域的理解

lydia：如何使用GPT解决本专业的一些问题，比如计算机专业写代码，文科专业查材料，excel统计，图像优化

1．安装部署参数高效微调环境，随机初始化一组预期收益率和协方差，计算并绘制资产的有效边界。

Step1: PEFT安装
由于LORA,AdaLORA都集成在PEFT上了，所以在使用的时候安装PEFT是必备项
方法一：PyPI
pip install peft
方法二：源码安装
pip install git+https://github.com/huggingface/peft
或者
git clone https://github.com/huggingface/peft
cd peft
pip install -e .
'''
# 示例：使用PEFT计算有效边界
import numpy as np
import peft
# # 假设有一组预期收益率和协方差
# returns = np.array([0.12, 0.18, 0.14])
# cov = np.array([
#   [0.1, 0.03, 0.05],
#   [0.03, 0.2, 0.06],
#   [0.05, 0.06, 0.15]
# ])
# # 使用PEFT创建有效边界
# ef = peft.EfficientFrontier(returns, cov)
# ef.plot()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#TODO: 调用 Helsinki-NLP/opus-mt-zh-en 模型实现 中文->英文 翻译
from transformers import MarianMTModel, MarianTokenizer
def Helsinki_NLP_opus_mt_zh_en():
    #TODO: C:\Users\86157\.cache\huggingface\hub
    # model_name = "Helsinki-NLP/opus-mt-zh-en"#opus-mt-en-fr
    model_name = "D:/data/PEFT/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    # Input text to translate
    # text = "Your text here"  # Replace with the text you want to translate
    text = "你的文本在这里"
    # Tokenize the text and generate translation
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"Translated text: {translated_text}")
    # Translated text: Here's your text.

'''
2．使用适配器微调大型语言模型的场景，假设你已经加载了预训练的BERT模型，现在想要添加一个适配器层用于微调情感分类任务。
请列出至少两个实现适配器微调的步骤，并提供相应的Python代码片段。
'''
import torch
#TODO: 步骤1：加载预训练的BERT模型
# BERT-base-uncased模型本身并不直接进行翻译，它是一个预训练的语言模型，通常用于特征提取。实际的翻译任务需要结合其他技术或模型来实现。
# 例如，可以使用BERT的输出作为特征，训练一个seq2seq模型来进行翻译。
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
def BERT_Classification():
    model_name='D:/data/PEFT/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    #TODO: BertForSequenceClassification 用于分类
    # model = BertForSequenceClassification.from_pretrained(model_name)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    # print('output:',output)
    # Tokenize the input and create tensor
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Run through the BERT model
    with torch.no_grad():
        outputs = model(**encoded_input)
    # Extract the last hidden states (embedding) for each token
    last_hidden_states = outputs.last_hidden_state# last_hidden_states: torch.Size([1, 12, 768])
    # print("last_hidden_states:", last_hidden_states)
    # print("Last hidden states shape:", last_hidden_states.shape)

# # 步骤2：添加适配器层
# num_labels = 2  # 假设情感分类任务有两个类别
# adapter_layer = torch.nn.Linear(model.config.hidden_size, num_labels)
# # 将适配器层添加到模型中
# model.add_adapter("sentiment_adapter", adapter_layer, adapter_type="text_task")
# # 设置适配器层的参数需要梯度更新
# for param in model.sentiment_adapter.parameters():
#     param.requires_grad = True

'''
3．使用前缀微调方法微调大型语言模型以执行中文翻译成英文的任务。假设你已经加载了预训练的T5模型，现在想通过前缀微调方法使其适应中文翻译成英文的任务。
原文本：“这是一个关于大模型微调的问题，希望通过前缀微调方法来适应中文翻译成英文的任务。”
'''
# 步骤1：加载预训练的T5模型
from transformers import T5Tokenizer, T5ForConditionalGeneration
# 定义翻译函数
def translate(tokenizer,model):
    input_sequence_2 = "My eyes fill you with love"
    input_sequence_3 = "He is pretty kind that I did not expect"
    input_sequence_4 = "Her eyes are blue"
    task_prefix = "translate English to French: "# 添加前缀
    input_sequences = [input_sequence_2]
    input_sequences.extend([input_sequence_3, input_sequence_4])
    # 3.使用分词器对目标进行分词
    encoding = tokenizer([task_prefix + sequence for sequence in input_sequences],
                         padding=True, return_tensors="pt").input_ids
    # 4.对刚生成的分词进行目标语言的生成工作
    outputs = model.generate(encoding)
    # 5.对生成的目标语言进行解码工作，就可得到目标语言的文本，并打印
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print('result: ',result)
    return result

#TODO: 示例使用
# 加载T5模型和分词器
def load_T5_translate():
    model_name = 'D:/data/PEFT/t5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # english_text = "Hello, how are you?"
    translate_results = translate(tokenizer,model)
    print('translate_results: ',translate_results)
    # “我的眼睛里充满了爱”、“他很好，出乎我的意料”、“你的眼睛是蓝色的”。
    # 法语translate_results:
    # ["Mes yeux vous remplissent d'amour", 'Il est assez gentil que je ne m’attendais pas à', 'Ses yeux sont bleus']


# 定义翻译函数
def translate_src(text, prefix="translate English to German"):
    # 2.加载预训练的T5模型和Tokenizer
    model_name = "D:/data/PEFT/t5-base"  # 你可以选择其他版本，如t5-small, t5-base、t5-large等
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print('model:', model)
    input_sequence_2 = "My eyes fill you with love"
    input_sequence_3 = "He is pretty kind that I did not expect"
    input_sequence_4 = "Her eyes are blue"
    task_prefix = "translate English to French: "# 添加前缀
    input_sequences = [input_sequence_2]
    input_sequences.extend([input_sequence_3, input_sequence_4])
    # 3.使用分词器对目标进行分词
    encoding = tokenizer([task_prefix + sequence for sequence in input_sequences],
                         padding=True, return_tensors="pt").input_ids
    # 4.对刚生成的分词进行目标语言的生成工作
    outputs = model.generate(encoding)
    # 5.对生成的目标语言进行解码工作，就可得到目标语言的文本，并打印
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print('result: ',result)
    # 解码生成翻译结果
    # tokenized_text = tokenizer.encode(text, prefix=prefix + " : ")
    # result = tokenizer.decode(
    #     model.generate(tokenized_text), skip_special_tokens=True
    # )
    return result

#TODO: 示例使用
# english_text = "Hello, how are you?"
# german_text = translate_src(english_text, prefix="translate English to German")
# print(german_text)  # 输出: Hallo, wie geht's dir?


#TODO:
# type(tokenized_text): <class 'list'>
# AttributeError: 'list' object has no attribute 'shape'
'''
4．使用LoRA微调大型语言模型以执行文本摘要任务。假设你已经加载了预训练的BERT模型，现在想通过LoRA微调使其适应文本摘要任务。
原文本：“自然语言处理（Natural Language Processing，NLP）是AI领域中与计算机和人类语言之间交互的研究。NLP的目标是使计算机能够理解、解释、生成人类语言，使计算机与人的交互更加自然。它涉及文本处理、语音处理、机器翻译等多个方面的任务。”使用LoRA微调方法，将上述文本进行摘要生成，提取出主要信息。
'''
# 步骤1：加载预训练的BERT模型
from transformers import BertTokenizer, BertForSequenceClassification
import torch
def load_BERT_LoRA_Summary():
    model_name='D:/data/PEFT/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    print('model:',model)
    # 步骤2：定义LoRA微调的适应器层
    num_classes = 1  # 摘要生成任务只有一个类别
    lora_adapter = torch.nn.Linear(model.config.hidden_size, num_classes)

    # 将适配器层添加到模型中
    model.lora_adapter = lora_adapter
    print('model with loRA:',model)
    # 设置适配器层的参数需要梯度更新
    for param in model.lora_adapter.parameters():
        param.requires_grad = True

    # 步骤3：生成文本摘要
    original_text = "自然语言处理（Natural Language Processing，简称NLP）是人工智能领域中与计算机和人类语言之间交互的研究。NLP的目标是使计算机能够理解、解释、生成人类语言，使计算机与人的交互更加自然。它涉及到文本处理、语音处理、机器翻译等多个方面的任务。"
    input_ids = tokenizer.encode(original_text, return_tensors='pt')

    # 针对摘要生成任务，设置标签（例如，1表示摘要，0表示非摘要）
    labels = torch.tensor([1])

    # 进行微调
    outputs = model(input_ids, labels=labels)
    print('outputs: ',outputs)


# D:/software/python\pycharmVenv\python37tf114\Lib\site-packages\peft/tuners
# D:/software/python\pycharmVenv\python37tf114\Lib\site-packages\datasets
# C:/Users/86157/.cache\huggingface\hub
# D:/data/PEFT/financial_phrasebank



if __name__ == '__main__':
    # Helsinki_NLP_opus_mt_zh_en()
    load_T5_translate()

