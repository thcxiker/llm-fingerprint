import re
import openai
from process_data import *
from utils import *
from model_list import *

from openai import OpenAI
from openai.types import Completion, CompletionChoice, CompletionUsage
openai.api_key = "your_openai_api_key" 
def     batch_classify_with_gpt(generated_texts, options, model="text-davinci-003", max_tokens=50, temperature=0.0):
    """
    Uses batch input method to classify multiple generated texts via the GPT API.
    
    Parameters:
        generated_texts (list): A list of generated texts to be classified.
        options (dict): A dictionary of options in the format {'A': 'Description of A', 'B': 'Description of B', ...}.
        model (str): The name of the GPT model to use.
        max_tokens (int): The maximum number of tokens for the response.
        temperature (float): The randomness of the response; set to 0.0 for deterministic results.
    
    Returns:
        list: Classification results for each text.
    """
    # Build the batch input prompt
    prompt = ""
    for idx, text in enumerate(generated_texts):
        prompt += f"This is generated text {idx + 1}: \"{text}\".\nBased on the content, determine which of the following options it is closest to:\n"
        for key, description in options.items():
            prompt += f"{key}. {description}\n"
        prompt += "Choose only one option from A, B, C, D, or E. You must select exactly one option and respond with only the letter of your choice (e.g., A, B, C, D, or E).\n\n"

    # Send the request
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Parse the response for each text
    answers = response.choices[0].text.strip().split("\n")
    return [answer.strip() for answer in answers]

def single_classify_with_gpt(generated_text, option, model="gpt-3.5-turbo", max_tokens=128, temperature=0.0):
    """
    Uses single input method to classify each generated text individually via the GPT API.
    
    Parameters:
        generated_texts (list): A list of generated texts to be classified.
        options (dict): A dictionary of options in the format {'A': 'Description of A', 'B': 'Description of B', ...}.
        model (str): The name of the GPT model to use.
        max_tokens (int): The maximum number of tokens for the response.
        temperature (float): The randomness of the response; set to 0.0 for deterministic results.
    
    Returns:
        list: Classification results for each text.
    """
    # print(generated_text)
    answers = []
    prompt = f"This is a generated text: \"{generated_text}\".\nBased on the content, determine which of the following options it is closest to:\n"
    prompt += option
    prompt += "Choose only one option from A, B, C, D, or E. You must select exactly one option and respond with only the letter of your choice (e.g., A, B, C, D, or E)."
# 使用 ChatCompletion API 进行分类
    # print(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", 
                   "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    # 提取并返回分类结果
    # print(response)
    answer_content = response.choices[0].message.content.strip()
    # print(answer_content)
    return answer_content


# 示例：使用 read_json 并调用 GPT 分类
def classify_prompts(input_file):
    # 从 JSON 文件中读取 prompts 和所有数据
    input_parent_dir = os.path.dirname(input_file)
    outputfile =  os.path.join(input_parent_dir,"processed_decoded_outputs.json")
    # if os.path.exists(outputfile):
    #     print(f"{outputfile} 已存在，跳过处理。")
    #     return  # 跳过文件处理
    # _, all_datas = read_json(outputfile)

    _, all_datas = read_json(input_file)
    options = []

    for data in all_datas:
        if data['prompt']:
            # 提取指定属性中单词之后的部分
            extracted_text = extract_after_word(data['prompt'], "Choices:")
            # 创建一个新的字典，包含原始数据和新的提取结果
            options.append(extracted_text)
    classifications = []
    unclassified= [ ]
    # 执行批量分类
    # classifications = batch_classify_with_gpt(all_datas['ouput'], options)
    for text,option in zip(all_datas,options):
        classification = single_classify_with_gpt(text["output"], option)
        match = re.match(r"([A-E])", classification)
        if match:
            # print(match.group(1))
            classifications.append(match.group(1))  # 只提取选项的字母
        else:
            print(classification)
            classifications.append(classification)
            unclassified.append(text["output"] + classification)
        
    print(classifications)
    # with open(outputfile, 'w', encoding='utf-8') as f:
    #     for data in all_datas:
    #         if data['classification'] not in {'A', 'B', 'C', 'D', 'E'}:
    #             print(data['classification'])
    #             data['classification'] = extract_option_letter(data['classification'] ) # 在原始数据中添加分类结果
    #         f.write(json.dumps(data) + '\n')

    with open(outputfile, 'w', encoding='utf-8') as f:
        print("outputfile写入中")

        # 将分类结果与原始数据组合
        for data, classification in zip(all_datas, classifications):
            data['classification'] = classification  # 在原始数据中添加分类结果
            f.write(json.dumps(data) + '\n')

    debugfile =  os.path.join(input_parent_dir,"debug.json")

    with open(debugfile, 'a', encoding='utf-8') as f:
        print("debugfile写入中")
    # 将分类结果与原始数据组合
        json.dump(unclassified, f, ensure_ascii=False, indent=4)







# 示例用法
if __name__ == "__main__":
    # 设置 HTTP 和 HTTPS 代理
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
# 使用环境变量中的 API 密钥初始化 OpenAI 客户端
    client = OpenAI(
        api_key= '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'


    )

    # try:
    #     client.fine_tuning.jobs.create(
    #         model="gpt-3.5-turbo",
    #         training_file="file-abc123",
    #     )
    # except openai.APIConnectionError as e:
    #     print("The server could not be reached")
    #     print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    # except openai.RateLimitError as e:
    #     print("A 429 status code was received; we should back off a bit.")
    # except openai.APIStatusError as e:
    #     print("Another non-200-range status code was received")
    #     print(e.status_code)
    #     print(e.response)


    # 输入文件路径

    # 执行分类
    input_dir="./wiki_dataset"
    for model in MODEL_LIST:
        print("Current model: {}".format(model))
        input = os.path.join(input_dir, os.path.basename(model)) + "/decoded_outputs.json"
        classify_prompts(input)
        

    # # 打印结果
    # for result in results:
    #     print(result)