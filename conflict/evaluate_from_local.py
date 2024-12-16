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

def load_model_HF(model_name):
    import os

    # 设置本地缓存目录
    cache_dir = "/mnt/data/haochuntang/hfllm_cache"  # 你可以修改为你想要的缓存路径
    os.makedirs(cache_dir, exist_ok=True)
    llm = LLM(model=model_name, gpu_memory_utilization=float(0.9),
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=max_model_length,
                trust_remote_code=True,
                download_dir=cache_dir,  # 添加缓存目录参数
                cache_dir=cache_dir)     # 某些版本可能使用 cache_dir)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return (llm, sampling_params), tokenizer

def load_model():


    
    model_path = os.path.abspath(args.model)  # 获取绝对路径
    model_path = args.model
    print("Checking model directory contents:")
    print(os.listdir(model_path))
    llm = LLM(model=model_path,gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=torch.cuda.device_count() ,
                max_model_len=max_model_length,
                download_dir=None,  # 指定为None表示使用本地模型
                 enable_lora=True  )  # 添加这个参数表明使用本地路径 
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return (llm, sampling_params), tokenizer


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


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res
def select_by_indicies(df,sampled_indices):
    res = []
    for each in df:
        if each["question_id"] in sampled_indices:
            res.append(each)



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


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"./initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


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


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
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
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
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

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong

@torch.no_grad()
def eval_cot_sample(subject, model, tokenizer, val_df, test_df, output_path,is_save):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
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

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    if is_save:
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
    
    selected_subjects = all_subjects

    rlt_list = {}
    for model in HF_MODEL_LIST:
        save_result_dir = os.path.join(
            save_hfresult_dir, "/".join(model)
        )
        rlt_list[model] = []
        model, tokenizer = load_model_HF(model)
        for subject in selected_subjects:
            output_path = os.path.join(save_result_dir, "{}.json".format(subject))
            test_df = select_by_category(full_test_df, subject)
            val_df = select_by_indicies(full_val_df,sampled_indices[subject])
            rlt_list[model] = eval_cot_sample(subject, model, tokenizer, val_df, test_df ,output_path,is_save)
    return rlt_list
def main():
    model, tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        # val_df = select_by_indicies(full_val_df,sampled_indices=)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path)
        # rlt_list = eval_cot_sample(subject, model, tokenizer, val_df, test_df)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--lora", "-l", type=str, default="/mnt/data/yuliangyan/instruction_tuning_models/qwen25-ft")
    # parser.add_argument("--model", "-m", type=str, default="/mnt/data/yuliangyan/Qwen/Qwen2.5-7B")  # Update with local path
    for model in MODEL_LIST:
        parser.add_argument('--model', type=str, help='模型文件路径', default=model)

        args = parser.parse_args()
        os.makedirs(args.save_dir, exist_ok=True)
        global_record_file = args.global_record_file
        save_result_dir = os.path.join(
            args.save_dir, "/".join(args_generate_path(args))
        )
        file_prefix = "-".join(args_generate_path(args))
        timestamp = time.time()
        time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
        file_name = f"{file_prefix}_{time_str}_summary.txt"
        summary_path = os.path.join(args.save_dir, "summary", file_name)
        os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
        os.makedirs(save_result_dir, exist_ok=True)
        save_log_dir = os.path.join(args.save_dir, "log")
        os.makedirs(save_log_dir, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                    file_name.replace("_summary.txt",
                                                                                        "_logfile.log"))),
                                    logging.StreamHandler(sys.stdout)])

        main()

