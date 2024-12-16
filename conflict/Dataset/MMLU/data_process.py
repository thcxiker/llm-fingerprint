import json
from collections import defaultdict

# 初始化按 subject 分组的字典
data_by_subject = defaultdict(list)

# 读取 data.jsonl 文件并解析每一行
with open('./data.jsonl', 'r') as file:
    for line in file:
        # 解析每一行的 JSON
        data = json.loads(line.strip())
        
        # 使用 subject 字段将数据分组
        subject = data['subject']
        data_by_subject[subject].append(data)

# 将每个 subject 的数据保存到单独的文件中
for subject, entries in data_by_subject.items():
    # 生成文件名（如 anatomy_test.jsonl）
    filename = f"{subject}.jsonl"
    with open(filename, 'a', encoding='utf-8') as f:
        for entry in entries:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')  # 每条记录独立成行写入

    print(f"数据已保存至 {filename}")