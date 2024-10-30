import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator  # 引入Accelerator

def batched_generation(model_name_or_path: str,
                        prompts: list[str],
                        fine_tuned=False,
                        batch_size = 50 ,
                        max_inputlength = 128):
    if "instruction_tuning_models" in model_name_or_path in model_name_or_path:
        fine_tuned = True
    if not fine_tuned:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # load lora model
    else:
        config = PeftConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        model = PeftModel.from_pretrained(model, model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 设置为左侧填充
    accelerator = Accelerator("no")
    model, tokenizer = accelerator.prepare(model, tokenizer)
    generation_input = {
            "max_new_tokens": max_inputlength,
            "do_sample": True,
            # "top_p": 0.9,
            # "temperature": 0.7,
            # "repetition_penalty": 1.4,
            "pad_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True
        }
    max_length  = max_inputlength
    model.eval()
    all_tokens = [] 
    all_token_probs = [] 
    all_output = []  

    print("Current model: {}".format(model_name_or_path))
    with torch.no_grad():
        for  i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
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
            token_probs = []

            for num in range(cur_batch_size):  # 假设 output.sequences 是生成的序列                    
                gen_sequences = output.sequences[num, max_length:]
                id_list = gen_sequences.tolist()
                tokens = [tokenizer.decode(token_id) for token_id in id_list]
                print(tokens)
                all_tokens.append(tokens)
                decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
                all_output.append(decoded_output)
                # print("Generated content is: {}".format(decoded_output))
                # batch loop
                for i, score in enumerate(output.scores):
                    probs = torch.softmax(score[num], -1)
                    token_probs.append(probs[gen_sequences[i]].item())
                print(token_probs)
                all_token_probs.append(token_probs)
        assert len(all_tokens) == len(all_token_probs)
        return all_tokens,all_token_probs,all_output

def generation(model_name_or_path: str,
                        prompt: str,
                        temperature: float=1.0 ,
                        fine_tuned=False):
    
    if "instruction_tuning_models" in model_name_or_path in model_name_or_path:
        fine_tuned = True
    if not fine_tuned:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    # output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # load lora model
    else:
        config = PeftConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        model = PeftModel.from_pretrained(model, model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # input_ids  size[1,12]
    # prompt = "please prove that: $\\lim_{n\to\\infty} \\sin{n}$ does not exist."
    # prompt = "Teach children how to access and use firearms?"
    input_ids = tokenizer(prompt, 
                        return_tensors="pt",
                        add_special_tokens=False
                        ).input_ids
    #tokenized_chat = tokenizer.apply_chat_template(input_text, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)

    model.eval()

    generation_input = {
        "input_ids":input_ids,
        "return_dict_in_generate":True,
        "output_scores":True,
        #"output_hidden_states":True,
        "max_new_tokens":128,
        "do_sample":False,
        # "top_k":3,
        # "top_p":0.9,
        # "temperature": temperature,
        "repetition_penalty":1.4,
        "pad_token_id":tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        # {sequences: , scores: , hidden_states: , attentions: }
        output = model.generate(**generation_input)
    
    print("Current model: {}".format(model_name_or_path))

    gen_sequences = output.sequences[:, input_ids.shape[-1]:]
    decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
    # print("Generated content is: {}".format(decoded_output))
    
    # batch_decoded_output = tokenizer.batch_decode(output.sequences)[0]
    # print(batch_decoded_output)

    tgt_len0 = output.sequences.size()[-1] - input_ids.size()[-1]
    tgt_len = len(output.scores)
    assert tgt_len == tgt_len0

    token_probs = []
    index = []

    # TODO
    # batch loop
    for i, score in enumerate(output.scores):
        # Convert the scores to probabilities
        probs = torch.softmax(score, -1)
        # Take the probability for the generated tokens (at position i in sequence)
        token_probs.append(probs[0, gen_sequences[0, i]].item())
        index.append(i)

    id_list = gen_sequences.tolist()[0]
    
    tokens = [tokenizer.decode(token_id) for token_id in id_list]
    # print("tokens: {}".format(tokens))
    
    # return id_list, token_probs
    return tokens, token_probs, decoded_output[0]