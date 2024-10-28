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

def inference(model_name, input_dir, out_dir):
    """
    使用指定模型对输入的 JSON 文件进行推理，并保存结果。
    
    :param model_name: str, 预训练模型的名称
    :param input_dir: str, 包含输入 JSON 文件的目录
    :param out_dir: str, 保存输出 JSON 文件的目录
    """
    
    # llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # 设置为左侧填充

    # sampling_params = SamplingParams(logprobs=20)
    
    # model = torch.nn.DataParallel(model)
    accelerator = Accelerator("no")
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # 获取输入目录下所有的 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # 创建输出目录路径，基于模型名称
    output_dir = os.path.join(out_dir, os.path.basename(model_name))
    # 如果输出目录不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输出目录下已经处理过的 JSON 文件
    out_files = glob.glob(os.path.join(output_dir, '*.json'))
    # 提取已经处理过的文件名列表
    processed_files = [os.path.basename(file) for file in out_files]

    # 遍历所有输入的 JSON 文件
    for input_file in tqdm(json_files, desc="Processing files"):
        # 如果当前文件已经处理过，则跳过
        # if os.path.basename(input_file) in processed_files:
        #     continue
        
        prompts = []
        all_datas = []
        
        # 打开并读取 JSON 文件中的每一行
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['prompt'])  # 提取并保存提示（prompt）
                all_datas.append(data)  # 保存所有数据
        tokenizer.pad_token = tokenizer.eos_token
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

        generation_input = {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.45,
            "repetition_penalty": 1.4,
            "pad_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True
        }
        batch_size = 20 # 调整为较小的批次
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
        print("Current model: {}".format(model_name))

        predictions = []
        pbs = []
        for prompt in prompts:
            inputs = tokenizer(prompt, 
                return_tensors="pt", 
                add_special_tokens=False,
                padding=True,  # 自动填充
            #    truncation=True  # 确保输入不会太长
                ).to("cuda")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            model.eval()
            generation_input = {
                "input_ids":input_ids,
                "attention_mask": attention_mask,  # 传递 attention_mask
                "return_dict_in_generate":True,
                "output_scores":True,
                #"output_hidden_states":True,
                "max_new_tokens":128,
                "do_sample":True,
                # "top_k":50,
                "top_p":0.9,
                "temperature":0.45,
                "repetition_penalty":1.4,
                # "eos_token_id":tokenizer.eos_token_id,
                # "bos_token_id":tokenizer.bos_token_id,
                "pad_token_id":tokenizer.eos_token_id,

            }

            with torch.no_grad():
                # {sequences: , scores: , hidden_states: , attentions: }
                output = model.generate(**generation_input)
            # print("shapeis",input_ids.shape[-1])
            gen_sequences = output.sequences[:, input_ids.shape[-1]:]
            decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
            print("Generated content is: {}".format(decoded_output))
            print("###################################################")
            
            # # batch_decoded_output = tokenizer.batch_decode(output.sequences)[0]
            # # print(batch_decoded_output)

            tgt_len0 = output.sequences.size()[-1] - input_ids.size()[-1]
            tgt_len = len(output.scores)
            assert tgt_len == tgt_len0
            print("Generated length is {}.".format(tgt_len0))
            print("###################################################")
            id_list = gen_sequences.tolist()[0]
            tokens = [tokenizer.decode(token_id) for token_id in id_list]
            print(id_list)
            print("tokens: {}".format(tokens))
            token_probs = []
            index = []
            logits = output.scores

            probs = [torch.softmax(log, dim=-1) for log in logits]
            # print(probs)
            # 获取生成文本的token ID和对应的概率
            generated_ids = output.sequences
            token_prob = 0
            candidate_logits = []

            for label in [" A", " B", " C", " D"]:
                token_prob = 0
                try:
                    # 对每个标签进行编码，获取 token id
                    label_ids = tokenizer.encode(label.strip(), add_special_tokens=False)
                    label_id = label_ids[-1]  # 获取最后一个 token ID（假设这是单个 token）
                    
                    # print(label_id,tokenizer.decode(label_id))
                    for i, token_id in enumerate(generated_ids[0][len(input_ids[0]):]):
                        # print(token_id.item() )
                        decoded_token =  tokenizer.encode(tokenizer.decode(token_id).replace(":", "").replace("(", "").strip(), add_special_tokens=False)
                        # print(decoded_token,tokenizer.decode(decoded_token.item()))
                        # print(decoded_token)
                        if decoded_token and decoded_token[-1] == label_id:
                            token_prob = probs[i][0, token_id].item()
                            # print(token_prob)
                            break
                    candidate_logits.append(token_prob)
                except Exception as e:
                    # 如果标签未找到，给出一个默认的低 logit 值
                    print(f"Exception encountered for label: {label}", flush=True)
                    print(f"Error message: {str(e)}", flush=True)
                    traceback.print_exc()  # 打印完整的堆栈跟踪信息
                    print(f"Warning: {label} not found. Artificially adding log prob of -100.", flush=True)
                    candidate_logits.append(0)
            # 将对数概率转换为概率分布

            candidate_logits = torch.tensor(candidate_logits).to(torch.float16)
            prob = torch.nn.functional.softmax(candidate_logits, dim=0).detach().cpu().numpy()
            # 根据最高概率的索引确定最终的预测答案
            print(candidate_logits)
            if len(candidate_logits) == 4:
                answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(prob)]
            else:
                print(f"Unexpected prob length: {len(candidate_logits)}, skipping.")
                answer = "Unknown"
            # answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(prob)]
            predictions.append(answer)
            # print(predictions)
            # 保存每个选项的概率
            pbs.append({'A': float(prob[0]), 'B': float(prob[1]), 'C': float(prob[2]), 'D': float(prob[3])})
                #     # 将输出结果保存到指定目录下
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        with open(output_file, 'a', encoding='utf-8') as f:
            # 每次写入 1000 条数据
            print(len(all_datas))
            print(len(predictions))

            # if(len(all_datas))
            for j in range(len(all_datas)):
                data = all_datas[j]
                data['prediction'] = predictions[j]
                data['prob'] = pbs[j]
                # 将修改后的数据写入文件
                f.write(json.dumps(data) + '\n')
# UNDO:将输入同意限制为一定长度然后
        # with torch.no_grad():
        #     for  in range(0, len(prompts), batch_size):
        #         batch_prompts = prompts[i:i + batch_size]
        #         inputs = tokenizer(batch_prompts, 
        #         return_tensors="pt", 
        #         add_special_tokens=False,
        #         padding=True,  # 自动填充
        #         # truncation=True  # 确保输入不会太长
        #         ).to("cuda")
        #         # inputs = accelerator.prepare(inputs)
        #         # input_ids = inputs["input_ids"]
        #         # attention_mask = inputs["attention_mask"]
        #         # if torch.cuda.is_available():
        #         #     input_ids = input_ids.to('cuda')
        #         #     attention_mask = attention_mask.to('cuda')      
        #         output = model.generate(**inputs, **generation_input)
        #         # gen_sequences = output.sequences[:, input_ids.shape[-1]:]
        #         # decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
        #         # print("Generated content is: {}".format(decoded_output))
        #         # 提取生成的序列
        #         generated_sequences = output.sequences
        #         # 生成的序列的形状为 (num, sequence_length)
        #         batch_size = generated_sequences.shape[0]  # 获取批次大小
        #         sequence_length = generated_sequences.shape[1]  # 生成序列的长度
        #         # 提取 scores（每个生成时间步的 logits）
        #         # for i, score in enumerate(output.scores):
        #         #     print(f"时间步 {i} 的 logits 大小: {score.size()}")
        #         predictions = []
        #         probs = []
        #         # print(scores.size())
        #     for num in range(len(output.sequences)):  # 假设 output.sequences 是生成的序列
        #         candidate_logits = []
        #         for label in ["A", "B", "C", "D"]:
        #             try:
        #                 # 对每个标签进行编码，获取 token id
        #                 label_ids = tokenizer.encode(label, add_special_tokens=False)
        #                 label_id = label_ids[0]  # 获取最后一个 token ID（假设这是单个 token）
        #                 # print(label_id)
        #                 # 提取每个时间步的 logits 或 scores
                        
        #                 # 遍历每个生成的 token 时间步
        #                 for score in enumerate(output.scores[-1]):
        #                         # Convert the scores to probabilities
        #                         probs = [torch.softmax(log, dim=-1) for log in score]
        #                         # Take the probability for the generated tokens (at position i in sequence)
        #                         print("PROBS:")
        #                         print(probs[0, ].item())
        #                         token_probs.append(probs[0, gen_sequences[0, i]].item())
        #                         index.append(i)
        #                         gen_sequences = output.sequences[:, input_ids.shape[-1]:]

        #                         id_list = gen_sequences.tolist()[0]
                            
        #                         for token_id, prob in zip(id_list, token_probs):
        #                             print(f"{tokenizer.decode(token_id)}: {prob}")
                            
        #                         tokens = [tokenizer.decode(token_id) for token_id in id_list]
        #                         print("tokens: {}".format(tokens))

        #             except Exception as e:
        #                 # 如果标签未找到，给出一个默认的低 logit 值
        #                 print(f"Exception encountered for label: {label}", flush=True)
        #                 print(f"Error message: {str(e)}", flush=True)
        #                 traceback.print_exc()  # 打印完整的堆栈跟踪信息
        #                 print(f"Warning: {label} not found. Artificially adding log prob of -100.", flush=True)
        #                 candidate_logits.append(-100)
        #         # 将对数概率转换为概率分布

        #         candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
        #         prob = torch.nn.functional.softmax(candidate_logits, dim=0).detach().cpu().numpy()
        #         # 根据最高概率的索引确定最终的预测答案
        #         if len(prob) == 4:
        #             answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(prob)]
        #         else:
        #             print(f"Unexpected prob length: {len(prob)}, skipping.")
        #             answer = "Unknown"
        #         # answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(prob)]
        #         predictions.append(answer)
        #         print(predictions)
        #         # 保存每个选项的概率
        #         probs.append({'A': float(prob[0]), 'B': float(prob[1]), 'C': float(prob[2]), 'D': float(prob[3])})
        #     # 将输出结果保存到指定目录下
        #     output_file = os.path.join(output_dir, os.path.basename(input_file))
        #     with open(output_file, 'a', encoding='utf-8') as f:
        #         # 每次写入 1000 条数据
        #         for j in range(len(all_datas)):
        #             data['prediction'] = predictions[j]
        #             data['prob'] = probs[j]
        #             # 将修改后的数据写入文件
        #             f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    # 设置命令行参数解析
    # parser = argparse.ArgumentParser(description="Run inference on input JSON files using a specified model.")
    # parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model.")
    # parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input JSON files.")
    # parser.add_argument('--out_dir', type=str, required=True, help="Directory to save the output JSON files.")
    # args = parser.parse_args()


    # 在某些耗内存的操作之后释放缓存
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary())

    # 调用推理函数
    # print("Your message", flush=True)

    for model in MODEL_LIST:
        # print("./conflict/"+model.split('/')[-1])
        # inference(model, "./conflictdata", "./conflictoutput/"+model.split('/')[-1])
        inference(model, "./normaldata", "./normaloutput/"+model.split('/')[-1])

