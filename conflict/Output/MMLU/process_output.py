import json
import os

def process_file(file_path):
    """
    处理单个 JSON 文件：在每条记录中添加 is_correct 字段，并输出处理后的 JSONL 文件。
    """
    output_data = []

    # 读取 JSONL 文件并解析内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            output = data['output']
            ori_a = data['answers'][0]
            print(output)
            print(ori_a)
            # 判断 output 是否包含正确答案
            is_correct = ori_a.replace(" ", "").strip().lower() in output.strip().lower()
            data["is_correct"] = is_correct  # 添加判断结果到数据中
            output_data.append(data)  # 将处理过的数据添加到输出列表

    # 生成输出文件路径
    output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_processed.jsonl"
    output_filepath = os.path.join(os.path.dirname(file_path), output_filename)

    # 将处理后的数据逐行写入到新的 JSONL 文件中
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for entry in output_data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"处理结果已保存至 {output_filepath}")
    
    # 返回处理过的数据
    return output_data




def process_all_files(folder_path):
    """
    遍历文件夹中的所有 JSON 文件，并对每个文件调用 process_file。
    """
# 使用 os.walk 递归遍历所有文件和子文件夹
    for dirpath, dirnames, filenames in os.walk(folder_path):
        
        for filename in filenames:
            # 检查文件是否为 .jsonl 文件
            print(filename)
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing file: {file_path}")
                # 调用 process_file 函数处理文件
                process_file(file_path)

    print("所有文件已处理")
if __name__ == '__main__':
    
    # 设置要处理的文件夹路径
    folder_path = './'  # 根据实际路径修改  
    # 执行批量处理  
    process_all_files(folder_path)
