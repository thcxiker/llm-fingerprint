import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.lora.request import LoRARequest
from model_list import *
from utils import *
from metrics import *
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df
def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df
def load_hf_model_TF(model_name_or_path, 
                  device='cuda' if torch.cuda.is_available() else 'cpu'
                  ):
    print("start to load model")
    try:
        cache_dir = "/mnt/data/haochuntang/hfllm_cache"  # 你可以修改为你想要的缓存路径

        # Load base model and tokenizer
        print(f"Loading base model '{model_name_or_path}' on {device}...")
        # load model for generation
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True,
                                                    cache_dir=cache_dir,  # 添加缓存目录参
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                    use_fast=False,
                                                    padding_side='left',
                                                    cache_dir=cache_dir
                                                    )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Model loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
def load_hf_model_VLM(model_name):
    # 设置本地缓存目录
    cache_dir = "/mnt/data/haochuntang/hfllm_cache"  # 你可以修改为你想要的缓存路径
    os.makedirs(cache_dir, exist_ok=True)
    llm = LLM(model=model_name, gpu_memory_utilization=float(0.9),
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=max_model_length,
                trust_remote_code=True,
                download_dir=cache_dir,  # 添加缓存目录参数
                )     # 某些版本可能使用 cache_dir)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return (llm, sampling_params), tokenizer
def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"./initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    # print("val_dfhere")
    # print(val_df)
    # val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    print("prompthere")

    # print(prompt)
    return prompt
def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    # print(prompt)
    return prompt

def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res
def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch,
                        sampling_params,
                    # lora_request=LoRARequest("lora_adapter", 1, args.lora)
)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch

def batch_generation(
                    model,
                    tokenizer,
                    prompt: list[str],
                    max_new_tokens: 2048,
                    ):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = 0

    # (batch_size, max_length)
    input_ids = tokenizer(
                        prompt, 
                        return_tensors="pt",
                        add_special_tokens=False,
                        padding=True,
                        ).input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)

    model.eval()

    generation_input = {
        "input_ids":input_ids,
        "return_dict_in_generate":True,
        "output_scores":True,
        "output_logits":True,
        #"output_hidden_states":True,
        "max_new_tokens":max_new_tokens,
        "do_sample":False,
        # "top_k":3,
        # "top_p":0.9,
        # "temperature": 0,
        # "repetition_penalty":1.4,
        # "pad_token_id":tokenizer.eos_token_id,
        "pad_token_id":0,
        # "stop":"Question:"
    }
    
    with torch.no_grad():
        # {sequences: , scores: , hidden_states: , attentions: }
        output = model.generate(**generation_input)

    gen_sequences = output.sequences[:, input_ids.shape[-1]:] # token_ids: (batch_size, max_gen_length)
    try:
        decoded_output = [tokenizer.decode(ids) for ids in gen_sequences] # texts: (batch_size, text_length))
    except Exception as e:
        decoded_output = ["" for _ in range(len(gen_sequences))]
        pass
    response_batch = []
    pred_batch = []
    print("output eg:",decoded_output[0])

    for output in decoded_output:
        generated_text = output
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch
    
    
def select_by_indicies(df,sampled_indices):
    res = []
    for each in df:
        if each["question_id"] in sampled_indices:
            res.append(each)
    return res

def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong
@torch.no_grad()
def eval_cot_sample(subject, model, tokenizer, val_df, test_df, output_path,is_save):
    # llm, sampling_params = model
    
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = 5
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)
    print("here start generate!!")
    # pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    pred_batch, response_batch = batch_generation(model,tokenizer,inference_batches,max_new_tokens)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    if is_save:
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        accu, corr, wrong = save_res(res, output_path)
        logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))



    res_list = get_res_list(res)

    return res_list
def get_res_list(res):
    rlt = []
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                rlt.append(each["question_id"])
                print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
                rlt.append(each["question_id"])

    return rlt
def evaluate_from_sample(sampled_indices, is_save):
    save_hfresult_dir = "./hug_rlt"
    if not os.path.exists(save_hfresult_dir):
        os.makedirs(save_hfresult_dir)
    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    print("all_subjects",all_subjects)
    selected_subjects = all_subjects

    rlt_list = {}

        # llm,SamplingParams = model
        # output = llm.generate("Hello, my name is")
        # print(output)
    # for subject in selected_subjects:
    #     # test_df = select_by_category(full_test_df, subject)
    #     # print(len(full_test_df))
    #     # print(len(full_val_df))

    #     # print(sampled_indices[subject])
    #     test_df = select_by_indicies(full_test_df,sampled_indices[subject])
    #     val_df = select_by_category(full_val_df,subject)
    #     print(len(test_df))
    #     print(len(val_df))
    #     print("ALLVALDF")
    #     print(val_df)
    #     for model_name in HF_MODEL_LIST:
    #         print("current model:"+model_name)
    #         model_path = model_name.split('/')[-1]
    #         print("---------path ")
    #         print(model_path)
    #         save_result_dir = os.path.join(
    #             save_hfresult_dir, (model_path)
    #         )
    #         print("---------savepath ")

    #         output_path = os.path.join(save_result_dir, "{}.json".format(subject))
    #         print(output_path)
    #         rlt_list[model_name] = []
    #         model, tokenizer = load_model_HF(model_name)
    #         rlt_list[model_name] = eval_cot_sample(subject, model, tokenizer, val_df, test_df ,output_path,is_save)
    
    
    for model_name in HF_MODEL_LIST:
            rlt_list[model_name] = []
            print("current model:"+model_name)
            model_path = model_name.split('/')[-1]
            print("---------path--------- ")
            print(model_path)
            save_result_dir = os.path.join(
                save_hfresult_dir, (model_path)
            )
            for subject in selected_subjects:
                test_df = select_by_indicies(full_test_df,sampled_indices[subject])
                val_df = select_by_category(full_val_df, subject)
                output_path = os.path.join(save_result_dir, "{}.json".format(subject))
                print("---------savepath--------- ")
                print(output_path)
                # model, tokenizer = load_hf_model_VLM(model_name)
                model, tokenizer = load_hf_model_TF(model_name)
                if model is None:
                    print("load faied")
                    continue
                rlt_list[model_name] = eval_cot_sample(subject, model, tokenizer, val_df, test_df ,output_path,is_save)
    return rlt_list
subject_list = [
        "biology","business","chemistry","economics","computer science","economics",
        "engineering","health","history","law","math","other","philosophy","physics","psychology"
    ]
def extract_answers( sampled_indices):
    """
    根据抽取的题目序号，收集各个模型在各个类别下对应题目的回答。
    
    返回格式：
    {
        'category1': {
            'question1': {
                'model1': '正确序号',
                'model2': '回答内容',
                ...
            },
            'question2': { ... },
            ...
        },
        'category2': { ... },
        ...
    }
    """
    output_path = "./results/"
    rlt = {}

    for model in MODEL_LIST:
        model_name = model.split("/")[-1]
        model_path = os.path.join(output_path, "{}/CoT/all".format(model_name))
        # print(model_path)
        # print(os.path.exists(model_path))
        data = read_answers(model_path)
        # print(data.keys())
        # print(data["math"])
        extract = read_corr_order(sampled_indices,data)
        rlt[model_name] = extract
    # print(rlt)
   
    return rlt
def read_answers(model_path):


    """
    从指定目录中读取一个模型和类别的答案，返回一个嵌套字典。
    
    返回格式：
    {
        'model1': {
            'category1': [
                {'question_id': 1, 'answer': '回答1', 'pred': 'A'},
                {'question_id': 2, 'answer': '回答2', 'pred': None},
                ...
            ],
            'category2': [ ... ],
            ...
        },

    }
    """

    categories = subject_list
    data = {}
    for category in categories:
        # print(category)
        output_path = os.path.join(model_path, "{}.json".format(category))

        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    entries = json.load(f)
                    # 确保每个条目包含question_id和answer
                    valid_entries = [e for e in entries if 'question_id' in e and 'answer' in e]
                    # print(f"有效的条目：{valid_entries[0]}")
                    data[category] = valid_entries
                except json.JSONDecodeError as e:
                    print(f"错误读取 {output_path}: {e}")
    # print(data.keys())
    return data
def sample_questions(default_model_file, sample_size):
    """
    对每个类别随机抽取指定数量的题目，返回一个字典。
    
    返回格式：
    {
        'category1': [question_id1, question_id2, ...],
        'category2': [question_id3, question_id4, ...],
        ...
    }
    """
    # seed=12345
    seed = random.randint(0, 2**32 - 1)

    data  = read_answers(default_model_file)
    random.seed(seed)
    sampled_indices = {}
    # 获取所有类别
    categories = subject_list

    for category in categories:
        # 确保所有模型在该类别下有相同数量的题目
        # output_path = os.path.join(default_model_file, "{}.json".format(category))
        num_questions = len(data.get(category, []))
        actual_sample_size = min(sample_size, num_questions)
        sampled = random.sample(range(num_questions), actual_sample_size)
        # 获取对应的question_ids
        if actual_sample_size > 0:
            # 假设所有模型在同一类别下的question_id一致
            sampled_question_ids = [data[category][idx]['question_id'] for idx in sampled]
            sampled_indices[category] = sampled_question_ids
    return sampled_indices

def get_sampled_questions(default_model_file,  sample_size):
    sampled_indices = sample_questions(default_model_file, sample_size)
    data = read_answers(default_model_file)
    output_data = {}
    for category, question_ids in sampled_indices.items():  
        output_data[category] = []
        for qid in question_ids:  # 遍历当前类别下的每个问题 ID
            for e in data[category]:  # 遍历 data 中的每个字典
                if e['question_id'] == qid:  # 如果问题 ID 匹配
                    output_data[category].append({'question_id': e['question_id'], 'question': e['question']})




    # Define the output temporary file path (using a temp file to prevent loss)
    temp_output_file = 'sampled_questions_temp.json'

    # Save the sampled questions to a temporary JSON file
    with open(temp_output_file, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    # Print the file path for verification
    print(f"Sampled questions saved to: {temp_output_file}")
    
    # Return the sampled indices (or other useful data)
    return output_data

def read_corr_order(sampled_indices,data):
    """
    从单个模型的 JSONL 文件中按类别计算准确率。
    :param json_file: JSONL 文件路径，包含每个记录的类别和正确性
    :return: 正确序号])) for model in data])
        actual_sample_size = min(sample_size, num_questions)
    """
    extracted = {}

    # 逐行读取 JSONL 文件
    # print(type(sampled_indices))  # 查看类型
    # print(sampled_indices)        # 打印内容，确认其值
    # print(data)  # 打印 data
    # for e in data:
    #     print(type(e), e)  # 打印每个元素的类型及内容

    for category, question_ids in sampled_indices.items():  
        extracted[category] = []  # 初始化每个类别对应的空列表
        for qid in question_ids:  # 遍历当前类别下的每个问题 ID
            for e in data[category]:  # 遍历 data 中的每个字典
                # print(e['pred'])  # 打印预测值
                # print(e['question_id'])  # 打印问题 ID
                if e['question_id'] == qid:  # 如果问题 ID 匹配
                    if e['pred'] == e['answer']:  # 如果预测值和答案相同
                        extracted[category].append(qid)  # 将问题 ID 添加到结果列表
    # print(extracted)
    return extracted

def evaluate_pairs(extracted_answers,victim,perpetrator):
    """
    评估配对的正确性，计算ROC AUC和配对准确率、精确率、召回率、F1分数。
    """
    # 遍历每个模型的数据
    extended_indices={}
    victim_indices = []
    for modelname, categories in extracted_answers.items():
        # 遍历每个类别
        if(modelname == victim):
            for category, indices in categories.items():
            # 遍历每个索引
                victim_indices.extend(indices)
        else: 
            extended_indices[modelname] = []  # Initialize an empty list for the key
            for category, indices in categories.items():
                # 遍历每个索引
                extended_indices[modelname].extend(indices)
    # print("victim",victim,"per",perpetrator)
    # print(extended_indices)
    # 计算 Jaccard 相似性并排序
    jaccard_results = {
        modelname: jaccard_similarity(victim_indices, indices)
        for modelname, indices in extended_indices.items()
    }
    # print(jaccard_results)
    # 按 Jaccard 相似性从大到小排序
    sorted_results = sorted(jaccard_results.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_results[0][0])
    return sorted_results[0][0] == perpetrator
def main(sample_size, type):
    """
    主函数：读取数据，抽样，提取回答，计算相似度，评估配对，展示结果。
    
    参数:
        sample_size (int): 每个类别抽取的题目数量。
    """
    # 配置路径

    default_model_file = './results/deepseek-chat-ft/CoT/all'  
    # 2. 抽取题目序号
    sampled_indices = sample_questions(default_model_file, sample_size)
    # sampled_question = get_sampled_questions (default_model_file,sample_size)
    # print(sampled_indices)
    # for category, indices in sampled_indices.items():
    #     print(f"类别 '{category}': {', '.join([str(idx+1) for idx in indices])}")


    if( type == "offline"):
    # 3. 根据抽取的题目序号收集回答
        extracted_answers = extract_answers(sampled_indices)
    elif (type == "online"):
        print("ONLINE")
        extracted_answers = evaluate_from_sample(sampled_indices,True)
        # print(extracted_answers)
        # extracted_answers = get_answers()
        return
    # Ensure you have a list of modelnames to work with
    Pos_pairs = 0
    modelnames = list(extracted_answers.keys())  # Get the list of all modelnames
    # print(modelnames)
    for i in range(0, len(modelnames) - 1, 2):  # Loop through modelnames in steps of 2
        modelname = modelnames[i]  # Current modelname
        next_modelname = modelnames[i + 1]  # Next modelname (i+1)

        # 4. Read the pairing information for the current model and the next one
        if  evaluate_pairs(extracted_answers, modelname, next_modelname):
            Pos_pairs+=1
    print(f"acc: {Pos_pairs / (len(modelnames)/2)}")
    print(Pos_pairs)
    
    print("\n所有类别处理完毕。")
    return Pos_pairs

if __name__ == "__main__":
    # 设置每个类别抽取的题目数量，例如每个类别抽取5个题目
    SAMPLE_SIZE = 5
    main(SAMPLE_SIZE,"online")
    # sum_pos = 0
    # all = 0
    # for i in range(0,100000,1):
    #     sum_pos += main(SAMPLE_SIZE, "offline" )
    #     all +=4
    # print(f"sum_acc: {sum_pos / all}")
