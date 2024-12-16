import json
import os
import torch
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from model_list import *
# from vllm import LLM, SamplingParams
import traceback  # 用于打印详细的异常信息
from accelerate import Accelerator  # 引入Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from process_data import *
def inference(model_name, input_dir, out_dir,all_datas=[],prompts =[]):
    """
    使用指定模型对输入的 JSON 文件进行推理，并保存结果。
    
    :param model_name: str, 预训练模型的名称
    :param input_dir: str, 包含输入 JSON 文件的目录
    :param out_dir: str, 保存输出 JSON 文件的目录
    """
    fine_tuned = False
    # prompts,all_datas = read_dataset("tau/commonsense_qa",100)
    print(os.path.basename(model_name))
    # 创建输出目录路径，基于模型名称
    output_dir = os.path.join(out_dir, os.path.basename(model_name))
    # 如果输出目录不存在则创建
    output_file = os.path.join(output_dir,"MMLU_5shots_outputs.json")
    print(output_file)
    # 检查 output_file 是否存在
    # if os.path.exists(output_file):
    #     print(f"{output_file} 已存在，跳过操作")
    #     return
    # else:
    #     print(f"{output_file} 不存在，继续执行操作")
    os.makedirs(output_dir, exist_ok=True)

    if "test" in model_name:
        fine_tuned = True
    if not fine_tuned:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        return_dict=True, 
                                                        device_map="auto",
                                                        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load lora model
    else:
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        model = PeftModel.from_pretrained(model, model_name, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    tokenizer.padding_side = "left"  # 设置为左侧填充

    
    # accelerator = Accelerator("no")
    # model, tokenizer = accelerator.prepare(model, tokenizer)

    tokenizer.pad_token = tokenizer.eos_token
    print("Current model: {}".format(model_name))

        # 批量生成输入
        
        # inputs = tokenizer(prompts, 
        #                 return_tensors="pt", 
        #                 add_special_tokens=False,
        #                 padding=True,  # 自动填充
        #                 # truncation=True  # 确保输入不会太长
        #                 )
        # # inputs = accelerator.prepare(inputs)
        # input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]
        # model.eval()

        # # 如果 GPU 可用，则将输入移到 GPU 上
        # if torch.cuda.is_available():
        #     input_ids = input_ids.to('cuda')
        #     attention_mask = attention_mask.to('cuda')      
        # 设置生成配置参数


        # for i in range(0, len(all_datas), batch_size):
            # batch_prompts = prompts[i:i + batch_size]
        #     output = llm.generate(batch_prompts, sampling_params)
        
            # predictions = []
            # probs = []
        #     for num in range(len(output)):
        #         candidate_logits = []
        #         for label in [" A", " B", " C", " D"]:
        #             try:
        #                 label_ids = tokenizer.encode(label, add_special_tokens=False)
        #                 label_id = label_ids[-1]
        #                 print("Labelid: {label_id} ")
        #                 if not label_ids:
        #                     print(f"Label {label} is not found in the tokenizer's vocabulary!")
        #                 candidate_logits.append(output[num].outputs[0].logprobs[0][label_id].logprob)
        #             except Exception as e:
        #                 print(f"Exception encountered for label: {label}", flush=True)
        #                 print(f"Error message: {str(e)}", flush=True)
        #                 traceback.print_exc()  # 打印完整的堆栈跟踪信息
        #                 print(f"Warning: {label} not found. Artificially adding log prob of -100.")
        #                 candidate_logits.append(-100)

        # predictions = []
        # pbs = []
        # for prompt in prompts:
        #     inputs = tokenizer(prompt, 
        #         return_tensors="pt", 
        #         add_special_tokens=False,
        #         padding=True,  # 自动填充
        #     #    truncation=True  # 确保输入不会太长
        #         ).to("cuda")
        #     input_ids = inputs["input_ids"]
        #     attention_mask = inputs["attention_mask"]
        #     model.eval()
        #     generation_input = {
        #         "input_ids":input_ids,
        #         "attention_mask": attention_mask,  # 传递 attention_mask
        #         "return_dict_in_generate":True,
        #         "output_scores":True,
        #         #"output_hidden_states":True,
        #         "max_new_tokens":128,
        #         "do_sample":True,
        #         # "top_k":50,
        #         "top_p":0.9,
        #         "temperature":0.45,
        #         "repetition_penalty":1.4,
        #         # "eos_token_id":tokenizer.eos_token_id,
        #         # "bos_token_id":tokenizer.bos_token_id,
        #         "pad_token_id":tokenizer.eos_token_id,

        #     }

        #     with torch.no_grad():
        #         # {sequences: , scores: , hidden_states: , attentions: }
        #         output = model.generate(**generation_input)
        #     # print("shapeis",input_ids.shape[-1])
        #     gen_sequences = output.sequences[:, input_ids.shape[-1]:]
        #     decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
        #     print("Generated content is: {}".format(decoded_output))
        #     print("###################################################")
            
        #     # # batch_decoded_output = tokenizer.batch_decode(output.sequences)[0]
        #     # # print(batch_decoded_output)

        #     tgt_len0 = output.sequences.size()[-1] - input_ids.size()[-1]
        #     tgt_len = len(output.scores)
        #     assert tgt_len == tgt_len0
        #     print("Generated length is {}.".format(tgt_len0))
        #     print("###################################################")
        #     id_list = gen_sequences.tolist()[0]
        #     tokens = [tokenizer.decode(token_id) for token_id in id_list]
        #     print(id_list)
        #     print("tokens: {}".format(tokens))
        #     token_probs = []
        #     index = []
        #     logits = output.scores

        #     probs = [torch.softmax(log, dim=-1) for log in logits]
        #     # print(probs)
        #     # 获取生成文本的token ID和对应的概率
        #     generated_ids = output.sequences
        #     token_prob = 0
        #     candidate_logits = []

        #     for label in [" A", " B", " C", " D"]:
        #         token_prob = 0
        #         try:
        #             # 对每个标签进行编码，获取 token id
        #             label_ids = tokenizer.encode(label.strip(), add_special_tokens=False)
        #             label_id = label_ids[-1]  # 获取最后一个 token ID（假设这是单个 token）
                    
        #             # print(label_id,tokenizer.decode(label_id))
        #             for i, token_id in enumerate(generated_ids[0][len(input_ids[0]):]):
        #                 # print(token_id.item() )
        #                 decoded_token =  tokenizer.encode(tokenizer.decode(token_id).replace(":", "").replace("(", "").strip(), add_special_tokens=False)
        #                 # print(decoded_token,tokenizer.decode(decoded_token.item()))
        #                 # print(decoded_token)
        #                 if decoded_token and decoded_token[-1] == label_id:
        #                     token_prob = probs[i][0, token_id].item()
        #                     # print(token_prob)
        #                     break
        #             candidate_logits.append(token_prob)
        #         except Exception as e:
        #             # 如果标签未找到，给出一个默认的低 logit 值
        #             print(f"Exception encountered for label: {label}", flush=True)
        #             print(f"Error message: {str(e)}", flush=True)
        #             traceback.print_exc()  # 打印完整的堆栈跟踪信息
        #             print(f"Warning: {label} not found. Artificially adding log prob of -100.", flush=True)
        #             candidate_logits.append(0)
        #     # 将对数概率转换为概率分布

        #     candidate_logits = torch.tensor(candidate_logits).to(torch.float16)
        #     prob = torch.nn.functional.softmax(candidate_logits, dim=0).detach().cpu().numpy()
        #     # 根据最高概率的索引确定最终的预测答案
        #     print(candidate_logits)
        #     if len(candidate_logits) == 4:
        #         answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(prob)]
        #     else:
        #         print(f"Unexpected prob length: {len(candidate_logits)}, skipping.")
        #         answer = "Unknown"
        #     # answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(prob)]
        #     predictions.append(answer)
        #     # print(predictions)
        #     # 保存每个选项的概率
        #     pbs.append({'A': float(prob[0]), 'B': float(prob[1]), 'C': float(prob[2]), 'D': float(prob[3])})
        #         #     # 将输出结果保存到指定目录下
        # output_file = os.path.join(output_dir, os.path.basename(input_file))
        # with open(output_file, 'a', encoding='utf-8') as f:
        #     # 每次写入 1000 条数据
        #     print(len(all_datas))
        #     print(len(predictions))

        #     # if(len(all_datas))
        #     for j in range(len(all_datas)):
        #         data = all_datas[j]
        #         data['prediction'] = predictions[j]
        #         data['prob'] = pbs[j]
        #         # 将修改后的数据写入文件
        #         f.write(json.dumps(data) + '\n')
# UNDO:将输入统一限制为一定长度然后
    generation_input = {
        "max_new_tokens": 10,
        "do_sample": "true",
        "top_p": 0.1,
        "temperature": 0.00001,
        # "repetition_penalty": 1.4,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        # "output_scores": True
    }
    max_length  = 320

    predictions = []
    pbs = []
    decoded_outputs = []
    batch_size = 5 # 调整为较小的批次
    model.eval()

    with torch.no_grad():
        for  i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            print(batch_prompts[0])
            inputs = tokenizer(batch_prompts, 
            return_tensors="pt", 
            add_special_tokens=False,
            padding="max_length",  # 确保填充到统一长度
            truncation=True,       # 确保输入不会太长
            max_length=max_length   # 设置统一的最大长度
            ).to("cuda")
            
            output = model.generate(**inputs, **generation_input)

            generated_sequences = output.sequences
            # 生成的序列的形状为 (num, sequence_length)
            cur_batch_size = generated_sequences.shape[0]  # 获取批次大小
            # predictions = []
            probs = []
            for num in range(cur_batch_size):  # 假设 output.sequences 是生成的序列                    
                gen_sequences = output.sequences[num, max_length:]
                # print(gen_sequences)
                decoded_output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_sequences]
                decoded_output = ''.join(decoded_output).rstrip()
                decoded_outputs.append(decoded_output)
                print("Generated content is: {}".format(decoded_output))
                # Choicelist = [("A", -100), ("B", -100), ("C", -100), ("D", -100)]
                # batch loop
                # for i, score in enumerate(output.scores):


                #     # Convert the scores to probabilities
                #     probs = torch.softmax(score[num], -1)
                #     # print(probs.size())
                #     # Take the probability for the generated tokens (at position i in sequence)
                #     cur_id = gen_sequences[i]
                #     # print(cur_id)0
                #     cur_token = tokenizer.decode(cur_id,add_special_tokens=False).replace(":", "").replace("(", "").strip()
                #     # cpr cur with "A B C D"
                #     for idx, (token, prob) in enumerate(Choicelist):
                #         if cur_token == token and prob == -100:
                #         # 获取当前token的概率
                #             Choicelist[idx] = (token, probs[cur_id].item())  # 更新prob
                #             break  # 跳出循环，防止重复处理该token
                #         # if cur_token == token :
                #         # # 获取当前token的概率
                #         #     Choicelist[idx] = (Choicelist[idx][0], Choicelist[idx][1] + probs[cur_id].item())
                #         #     break  # 跳出循环，防止重复处理该token
                # # 提取 Choicelist 中非 None 的概率
                # candidate_logits = [prob  for _, prob in Choicelist]

                # # 转换为 tensor 进行 softmax 归一化
                # candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
                # softmax = torch.nn.functional.softmax(candidate_logits, dim=0).detach().cpu().numpy()

                # # 找出最大概率的选项
                # if len(softmax) == 4:
                #     answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(softmax)]
                # else:
                #     print(f"Unexpected prob length: {len(softmax)}, skipping.")
                #     answer = "Unknown"

                # print("Predicted Answer:", answer)
                # predictions.append(answer)
                # print(predictions)
                # 保存每个选项的概率
                # 将概率分布保存到字典中

                # probs = {'A': float(softmax[0]), 'B': float(softmax[1]), 'C': float(softmax[2]), 'D': float(softmax[3])}
                # pbs.append(probs)
                # print("Probability Distribution:", probs)



    # output_file = os.path.join(output_dir,"MMLU_outputs.json")
    # print(output_file)
    # # 检查 output_file 是否存在
    # if os.path.exists(output_file):
    #     print(f"{output_file} 已存在，跳过操作")
    # else:
    #     print(f"{output_file} 不存在，继续执行操作")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 每次写入 1000 条数据
        for j in range(len(all_datas)):
            data = all_datas[j]
            # data['prediction'] = predictions[j]
            # data['prob'] = pbs[j]
            
            # data.prob = pbs[j]
            # data.output = decoded_outputs[j]
            # data_dict = data.__dict__  # This converts the data object to a dictionary
            data['output'] = decoded_outputs[j]
            # 将修改后的数据写入文件
            f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    input_dir = "./Dataset/MMLU"
    output_dir = "./Output/MMLU"
    # 获取输入目录下所有的 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    alldatas=[]
    prompts=[]
    # # 获取输出目录下已经处理过的 JSON 文件
    # out_files = glob.glob(os.path.join(output_dir, '*.json'))
    # # # 提取已经处理过的文件名列表
    # processed_files = [os.path.basename(file) for file in out_files]
    # for input_file in tqdm(json_files, desc="Processing files"):
    #     all_data,prompt = add_few_prompts(path = input_file,sample_size= 20)
    #     alldatas.extend(all_data)
    #     prompts.extend(prompt)
    # # 保存 `alldatas` 到本地文件
    # with open("alldatas.json", "w", encoding="utf-8") as f:
    #     json.dump(alldatas, f, ensure_ascii=False, indent=4)

    # 保存 `prompts` 到本地文件
    # with open("prompts.json", "w", encoding="utf-8") as f:
    #     json.dump(prompts, f, ensure_ascii=False, indent=4)
    # 读取 `alldatas`
    with open("alldatas.json", "r", encoding="utf-8") as f:
        alldatas = json.load(f)

    # 读取 `prompts`
    with open("prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    for model in MODEL_LIST:
        inference(model, input_dir, output_dir,alldatas,prompts)
        # inference(model, "./test", "./testoutput/"+model.split('/')[-1])

