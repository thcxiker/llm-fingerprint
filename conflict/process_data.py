from dataclasses import dataclass
import glob
import json
import os

from tqdm import tqdm

from datasets import load_dataset

def read_json(input_file, atr = "prompts"):

    
        
    prompts = []
    all_datas = []
    
    # 打开并读取 JSON 文件中的每一行
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data[atr])  # 提取并保存提示（prompt）
            all_datas.append(data)  # 保存所有数据
    return prompts, all_datas
def read_dataset(dataset_name, sample_size=100):
    # 定义一个数据类，每个属性对应数据集中的一个字段
    @dataclass
    class ExampleData:
        prompt: str
        true_label: str
        

    # 从 Hugging Face 加载数据集
    dataset = load_dataset(dataset_name)
    
    # 初始化存储的容器
    dataset_prompts = []
    dataset_all_data = []
    dataset = dataset.rename_column("question", "Qustion") #original_column to Qustion
    dataset = dataset.rename_column("choices", "Choices")
    dataset = dataset.rename_column("answerKey", "Answers")
    # 打乱数据集并获取前 sample_size 条记录

    dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))

    prompt_prefix = """According to the your knowledge , choose the best option from the following list and respond with only A, B, C, or D. Do not generate any additional content beyond selecting one of these options.\nQuestion:"""
    prompt_post = """Choices:\n"""

    for example in tqdm(dataset,desc=f"Processing "):
        # choices_str = ", ".join(choice for choice,text in example['Choices'].values())
        choices_str = ", ".join(
            f"{label}: {text}" 
            for label, text in zip(example['Choices']['label'], example['Choices']['text'])
        )
        cur_prompt = prompt_prefix + example['Qustion'] + prompt_post + choices_str
        dataset_prompts.append(prompt_prefix + example['Qustion'] + prompt_post + choices_str)
        example_data = ExampleData(
        prompt=cur_prompt,
        true_label=example["Answers"])
        dataset_all_data.append(example_data)           # 保存所有数据

    all_prompts = dataset_prompts 
    all_data = dataset_all_data 

    return all_prompts, all_data
import json
import random

import json
import random



def read_jsonl(path, sample_size=None):
    """
    从 JSONL 文件读取数据，生成知识点，并确保 data 和 sampled_data 数量一致。
    
    参数:
    - path: JSONL 文件的路径
    - sample_size: 随机抽取的样本数量（默认为 None，即返回所有数据）
    
    返回:
    - data: 包含所有数据的列表
    - sampled_data: 包含与 data 一一对应的列表，每个元素是 (json_data, one_knowledge) 的元组
    """
    all_data = []

    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)

            # 为每个 json_data 生成一个唯一的知识点
            if 'questions' in json_data and 'answers' in json_data:
                question =json_data['questions']
                # answer = random.choice(json_data['answers'])
                one_knowledge = question 
                all_data.append((json_data, one_knowledge))  # 存储为 (json_data, one_knowledge) 元组

    # 随机抽取 sample_size 个样本
    if sample_size is not None and sample_size < len(all_data):
        sampled_data = random.sample(all_data, sample_size)
    else:
        sampled_data = all_data  # 如果 sample_size 为空或超过总数量，返回全部数据

    # 提取 json_data 和 knowledge 组成的列表
    data = [item[0] for item in sampled_data]
    sampled_data = [item[1] for item in sampled_data]

    return data, sampled_data

def add_few_prompts(path, sample_size=None, num_prompts=5):
    """
    从 JSONL 文件读取数据，生成多个 prompt，每个数据项附加 5 个不同序号的 prompt。
    
    参数:
    - path: JSONL 文件的路径
    - sample_size: 随机抽取的样本数量（默认为 None，即返回所有数据）
    - num_prompts: 为每个条目生成的 prompt 数量（默认为 5）
    
    返回:
    - data: 包含所有数据的列表
    - prompts_data: 包含与 data 一一对应的列表，每个元素是 (json_data, prompts) 的元组
    """

    data = []
    all_data = []
    seen_idx = set()
    file_name = os.path.basename(path)
    
    # 分割文件名，取 "test" 前的部分
    test_name = file_name.split("test")[0].rstrip("_")
    prompt_prefix ="Answer questions about "+ test_name
    # 读取 JSONL 文件
    with open(path, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == num_prompts:
                break
            json_data = json.loads(line)  # 解析每一行 JSON
            idx = json_data['idx']
            seen_idx.add(idx)
            prompt_prefix += "Q:"+json_data['questions'][0]+"A:"+json_data['answers'][0]+"\n"
        
        # print(prompt_prefix)

        for line in f:
            json_data = json.loads(line)
            # 假设每条数据中有 'idx' 字段作为唯一标识符
            idx = json_data['idx']
            if idx in seen_idx:
                continue  # 如果 idx 已经出现过，跳过当前条目
            seen_idx.add(idx)

            data.append(json_data)

            # 为每个 json_data 生成一个唯一的知识点
            if 'questions' in json_data and 'answers' in json_data:
                question = json_data['questions'][0]
                one_knowledge = prompt_prefix + question 
                all_data.append((json_data, one_knowledge))  # 存储为 (json_data, one_knowledge) 元组


    # 随机抽取 sample_size 个样本
    if sample_size is not None and sample_size < len(all_data):
        sampled_data = random.sample(all_data, sample_size)
    else:
        print(path+ "lackdata")
        sampled_data = all_data  # 如果 sample_size 为空或超过总数量，返回全部数据

    # 提取 json_data 和 prompts 组成的列表
    data = [item[0] for item in sampled_data]
    prompts_data = [item[1] for item in sampled_data]
    # print("Eg:"+prompts_data[0])
    return data, prompts_data


def save_json(data_list, output_file):
    """
    将列表中的数据逐行写入到指定的 JSONL 文件中。
    
    参数:
    - data_list: 要保存的数据列表，每个元素是一个字典
    - output_file: 输出文件的路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in data_list:
            # 将每个数据项转化为 JSON 格式字符串，并写入文件
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')
