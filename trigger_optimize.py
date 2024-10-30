from itertools import chain
import json
import os
import random
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from model_list import *
from metrics import *
from generation import *
from tqdm import tqdm
# 假设有一个列表包含所有模型
def compute_all_models(models, prompts, output_dir="./model_outputs_4_optimize"):
    """
    对于每个模型计算生成的 tokens，并评估模型之间的相似性，将结果输出到文件。
    
    :param models: list, 包含所有模型对象的列表
    :param prompt: str, 输入的 prompt
    :param output_dir: str, 输出文件的目录路径
    """
    os.makedirs(output_dir, exist_ok=True)

    for model_idx, model in enumerate(models):
        # 创建文件名，每个模型单独一个文件
        file_path = os.path.join(output_dir, os.path.basename(model))
        # 打开文件写入模式
        # 获取模型生成的 tokens 和概率
        tokens, token_probs, texts = batched_generation(model, prompts)
        with open(file_path, "w") as f:
            data = {}
            for j in range(len(tokens)):
                data['token'] = tokens[j]
                data['prob'] = token_probs[j]
                # 将修改后的数据写入文件
                f.write(json.dumps(data) + '\n')



    print(f"所有模型的输出已分批保存到目录 '{output_dir}' 中。")


# Hypothetical metric functions
def compute_intra_model_similarity(model, fine_tuned_model, prompt):
    """
    Calculates similarity between model and fine-tuned model outputs for a given prompt.
    """
    tokens1, token_probs1, text1 = generation(model, prompt)
    tokens2, token_probs2, text2 = generation(fine_tuned_model, prompt)
    
    res = jaccard_similarity(tokens1, tokens2)
    
    return res


def read_tokens_from_file(filename):
    """
    读取文件中的 tokens 并返回一个列表
    """
    tokens_list = []
    with open(filename, "r") as file:
        for line in file:
            data = json.loads(line)
            tokens_list.append(data['token'])  # 假设文件中每行包含 "token" 字段
    return tokens_list
def read_ith_token_from_file(filename, i):
    """
    从文件中读取第 i 行的 token。
    
    :param filename: str, 文件路径
    :param i: int, 想要读取的行的索引（从 0 开始）
    :return: 第 i 行的 token 的值，如果超出范围则返回 None
    """
    with open(filename, "r") as file:
        for line_num, line in enumerate(file):
            if line_num == i:
                data = json.loads(line)
                return data.get('token')  # 假设每行的 JSON 对象中有 "token" 字段
    # 如果文件中没有第 i 行，则返回 None
    return None
def compute_intra_model_similarity_alltks(model,fine_tuned_model):
    """
    Calculates similarity between model and fine-tuned model outputs for all prompts.
    """
    # add the output path here
    model = os.path.join("./model_outputs_4_optimize", os.path.basename(model))
    fine_tuned_model = os.path.join("./model_outputs_4_optimize", os.path.basename(fine_tuned_model))

    tokens1 = read_tokens_from_file(model)
    tokens2 = read_tokens_from_file(fine_tuned_model)
    res = []
    for i in range( len(tokens1)):
        res.append(jaccard_similarity(tokens1[i], tokens2[i]))

    return res
def compute_similarity_byidx(model1,model2,index):
    """
    Calculates similarity between model and fine-tuned model outputs for a prompt by index.
    """
    # add the output path here
    model1 = os.path.join("./model_outputs_4_optimize", os.path.basename(model1))
    model2 = os.path.join("./model_outputs_4_optimize", os.path.basename(model2))

    tokens1 = read_ith_token_from_file(model1, index)
    tokens2 = read_ith_token_from_file(model2, index)
   

    return jaccard_similarity(tokens1=tokens1 ,tokens2= tokens2)
def ALL_compute_intra_model_similarity(models, fine_tuned_models):
    """
    Calculates similarity between model and fine-tuned model outputs for all models.
    """
    # 初始化累加列表，假设每个输出列表的长度相同
    total_scores = None
    
    for model, fine_tuned in zip(models, fine_tuned_models):
        scores = compute_intra_model_similarity_alltks(model, fine_tuned)
        
        # 初始化 total_scores 列表的长度
        if total_scores is None:
            total_scores = [0] * len(scores)
        
        # 按索引累加每个模型对的分数
        for i in range(len(scores)):
            total_scores[i] += scores[i]
    
    # 将累加的结果除以模型的数量
    total_scores = [score / len(models) for score in total_scores]
    print(len(total_scores))
    
    return total_scores
    


def compute_inter_model_divergence(model1, model2, prompt):
    """Calculates divergence between two different models' outputs for a given prompt."""
    tokens1, token_probs1, text1 = generation(model1, prompt)
    tokens2, token_probs2, text2 = generation(model2, prompt)
    
    res = jaccard_similarity(tokens1, tokens2)
    
    return res

def ALL_compute_inter_model_divergence(models, subset_size, prompts_size):
#  """
#     随机选择模型对并从文件中提取 tokens然后计算模型之间的差异。
    
#     :param file_paths: list, 包含每个模型文件路径的列表
#     :param subset_size: int, 要采样的模型对数量
#     :param prompt: str, 提供给每个模型的输入（在此情况下可能不需要直接使用）
#     """
    # 随机采样模型对
    res = []
    # add the output path
    models = [os.path.join("./model_outputs_4_optimize", os.path.basename(model)) for model in models]
    for idx in range(prompts_size):
        model_pairs = random.sample([(i, j) for i in range(len(models)) for j in range(i + 1, len(models))], subset_size)
        # 计算所有采样的模型对的差异分数
        inter_div_score = sum(
            jaccard_similarity(read_ith_token_from_file(models[i],idx), read_ith_token_from_file(models[j],idx))
            for i, j in model_pairs
        ) / subset_size
        res.append(inter_div_score)
    print(len(res))
    return res

def optimize_prompts_with_pair_sampling(trigger_set, 
                                        models, 
                                        fine_tuned_models,
                                        M=10, 
                                        sample_size=100, 
                                        alpha=1.0, 
                                        beta=1.0, 
                                        subset_size=5):
    """
    Finds M optimized prompts from trigger_set by maximizing intra-model similarity and inter-model divergence with pair sampling.

    Parameters:
    - trigger_set: List of all prompts.
    - models: List of base models.
    - fine_tuned_models: Corresponding fine-tuned models.
    - M: Number of optimized prompts to find.
    - sample_size: Number of prompts to sample for evaluation.
    - alpha: Weight for intra-model similarity.
    - beta: Weight for inter-model divergence.
    - subset_size: Number of model pairs to sample for inter-model divergence.

    Returns:
    - List of optimized prompts.
    """
    prompt_weights = []
    
    # Calculate importance weights for each prompt
    print(f"start to initial search")
    intra_sim_score = ALL_compute_intra_model_similarity(models=models,fine_tuned_models=fine_tuned_models)
    inter_div_score = ALL_compute_inter_model_divergence(models=models,subset_size=subset_size,prompts_size=len(trigger_set))
    if len(intra_sim_score) != len(inter_div_score):
        print(len(intra_sim_score))
        print(len(inter_div_score))
        raise ValueError("intra_sim_score 和 inter_div_score 的长度不匹配")
    prompt_weights = [
            max(alpha * score - beta * div_score, 0) 
            for score, div_score in zip(intra_sim_score, inter_div_score)]    
    total_weight = sum(prompt_weights)
    prompt_probs = [weight / total_weight for weight in prompt_weights]
    print(prompt_probs)
    # return
    # for data in tqdm(trigger_set):
    #     # print(data)
    #     prompt = data['prompt']
    #     # intra_sim_score = sum(
    #     #     compute_intra_model_similarity(model, fine_tuned, prompt) 
    #     #     for model, fine_tuned in zip(models, fine_tuned_models)
    #     # )
    #     intra_sim_score = sum(
    #         compute_intra_model_similarity(model, fine_tuned, prompt) 
    #         for model, fine_tuned in zip(models, fine_tuned_models)
    #     ) / len(models)
        
    #     # Randomly sample subset_size pairs of models for inter-model divergence
    #     model_pairs = random.sample([(i, j) for i in range(len(models)) for j in range(i + 1, len(models))], subset_size)
    #     inter_div_score = sum(
    #         compute_inter_model_divergence(models[i], models[j], prompt) 
    #         for i, j in model_pairs
    #     ) / subset_size
        
    #     # Importance weight based on prompt characteristics
    #     weight = max(alpha * intra_sim_score - beta * inter_div_score, 0)
    #     prompt_weights.append(weight)

    # Normalize weights to form a probability distribution
    # total_weight = sum(prompt_weights)
    # prompt_probs = [weight / total_weight for weight in prompt_weights]

    # Sample prompts based on their importance weights
    candidate_indices = random.choices(range(len(trigger_set)), weights=prompt_probs, k=sample_size)
    prompt_scores = []

    for idx in candidate_indices:
        intra_sim_score = 0
        inter_div_score = 0

        # Compute intra-model similarity
        for model, fine_tuned in zip(models, fine_tuned_models):
            intra_sim_score += compute_similarity_byidx(model, fine_tuned, idx)
        
        # Sample model pairs dynamically for inter-model divergence
        model_pairs = random.sample([(i, j) for i in range(len(models)) for j in range(i + 1, len(models))], subset_size)
        inter_div_score += sum(
            compute_similarity_byidx(models[i], models[j], idx) 
            for i, j in model_pairs
        )

        # Calculate overall score
        score = alpha * intra_sim_score - beta * inter_div_score
        prompt_scores.append((idx, score))

    # Sort and select the top M prompts
    optimized_prompts = sorted(prompt_scores, key=lambda x: x[1], reverse=True)[:M]
    indices = [index for index, score in optimized_prompts]
    
    # 根据 indices 从原始数据集中提取对应的 prompts
    selected_data = trigger_set.select(indices)

    # 将结果保存到新的数据集
    selected_data.save_to_disk('./data/optimized_trigger_set')
    # 如果需要返回提取的 prompts
    prompts = selected_data['prompt']  # 提取所有 `prompt` 字段的内容

    # 返回提取的 prompts 列表
    return prompts    
    # candidate_prompts = random.choices(trigger_set["prompt"], weights=prompt_probs, k=sample_size)
    # prompt_scores = []

    # print(f"start to search the final trigger set")
    # for prompt in tqdm(candidate_prompts):
    #     intra_sim_score = 0
    #     inter_div_score = 0

    #     # Compute intra-model similarity
    #     for model, fine_tuned in zip(models, fine_tuned_models):
    #         intra_sim_score += compute_intra_model_similarity(model, fine_tuned, prompt)
        
    #     # Sample model pairs dynamically for inter-model divergence
    #     model_pairs = random.sample([(i, j) for i in range(len(models)) for j in range(i + 1, len(models))], subset_size)
    #     inter_div_score += sum(
    #         compute_inter_model_divergence(models[i], models[j], prompt) 
    #         for i, j in model_pairs
    #     )

    #     # Calculate overall score
    #     score = alpha * intra_sim_score - beta * inter_div_score
    #     prompt_scores.append((prompt, score))

    # # Sort and select the top M prompts
    # optimized_prompts = sorted(prompt_scores, key=lambda x: x[1], reverse=True)[:M]
    # prompts = [prompt for prompt, score in optimized_prompts]
    
    # data = {"prompt": prompts}
    # dataset = Dataset.from_dict(data)
    # dataset.save_to_disk('./data/optimized_trigger_set')
    
    # return prompts

if __name__ == '__main__':
    # Example usage
    # answer_set = load_from_disk("./data/optimized_trigger_set")
    # for item in answer_set:
    #     print(item)

    # print(answer_set.select(range(5)))
    seed_trigger_set = load_from_disk("./data/seed_trigger_set")
    print(len(seed_trigger_set))
    # seed_trigger_set = seed_trigger_set.select(range(20, 40))
    # seed_trigger_set = seed_trigger_set[0 : 20]
    models = [
        "/mnt/data/yuliangyan/yuliangyan/meta-llama/Meta-Llama-3-8B",
        "/mnt/data/yuliangyan/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct",
        "/mnt/data/yuliangyan/yuliangyan/meta-llama/Meta-Llama-3.1-8B",
        "/mnt/data/yuliangyan/yuliangyan/mistralai/Mistral-7B-v0.1",
        "/mnt/data/yuliangyan/yuliangyan/deepseek-ai/deepseek-llm-7b-base",
        "/mnt/data/yuliangyan/yuliangyan/deepseek-ai/deepseek-llm-7b-chat",
        # "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B",
        # "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct",
    ]
    fine_tuned_models = [
       "/home/haochuntang/instruction_tuning_models/llama3-ft",
        "/home/haochuntang/instruction_tuning_models/llama3-instruct-ft",
        "/home/haochuntang/instruction_tuning_models/llama31-ft",
        "/home/haochuntang/instruction_tuning_models/mistral-ft",
        "/home/haochuntang/instruction_tuning_models/deepseek-ft",
        "/home/haochuntang/instruction_tuning_models/deepseek-chat-ft",
    ]
    # compute_all_models(models=models+fine_tuned_models,prompts=seed_trigger_set["prompt"],)

    # Find optimized prompts
    optimized_prompts = optimize_prompts_with_pair_sampling(seed_trigger_set, 
                                                            models, 
                                                            fine_tuned_models, 
                                                            M=15, 
                                                            sample_size=150, 
                                                            alpha=0.4, 
                                                            beta=0.6,
                                                            subset_size=2
                                                            )
    # print("Optimized prompts:", optimized_prompts)