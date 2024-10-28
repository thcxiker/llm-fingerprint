import pandas as pd


Confilict_template = """According to the your knowledge and evidence provided, choose the best option from the following list and respond with only A, B, C, or D. Do not generate any additional content beyond selecting one of these options.
Evidence: {evidence}
Question: {question}
Choices:
A) {choice_1}
B) {choice_2}
C) {choice_3}
D) {choice_4}
"""
# Function to generate prompts based on the DataFrame
def generate_conflict(row, kind):

    return Confilict_template.format(
            evidence=row[kind],
            question = row['question'],
            choice_1=row['options'][0],
            choice_2=row['options'][1],
            choice_3=row['options'][2],
            choice_4=row['options'][3],

        )
Normal_template = """According to the your knowledge , choose the best option from the following list and respond with only A, B, C, or D. Do not generate any additional content beyond selecting one of these options.
Question: {question}
Choices:
A) {choice_1}
B) {choice_2}
C) {choice_3}
D) {choice_4}
"""
# Function to generate prompts based on the DataFrame
def generate_normal(row):

    return Normal_template.format(
            question = row['question'],
            choice_1=row['options'][0],
            choice_2=row['options'][1],
            choice_3=row['options'][2],
            choice_4=row['options'][3],

        )


if __name__ == "__main__":
    df = pd.read_parquet('./query_result.parquet')
    # 假设你已经有 true_label, replaced_label, 和 category 列
    df['correct_option'] = df['correct_option']
    df['replaced_option'] = df['replace_option']
    df['uncertain_option'] = df['uncertain_option']

    # 创建一个空的 DataFrame，用于存储每个类别生成的 prompt 和 label
    results = pd.DataFrame()
    df['misinformation_conflict_evidence'] = df['misinformation_conflict_evidence_evidence'] 
    # 需要生成的类别列表
    category_list = ["semantic", "temporal", "misinformation"]

    # 对每一行应用多个类别
    for category in category_list:
        # 生成当前类别的 prompt
        df['prompt'] = df.apply(generate_conflict, axis=1, kind=category+"_conflict_evidence")
        
        # 创建包含当前类别信息的 DataFrame
        temp_df = pd.DataFrame({
            'category': category,
            'prompt': df['prompt'],
            'true_label': df['correct_option'],
            'replaced_label': df['replaced_option'],
            'uncertain_label': df['uncertain_option']
        })

        # 将当前类别的结果追加到总结果中
        results = pd.concat([results, temp_df], ignore_index=True)

    # 保存结果为 CSV 和 JSON
    results.to_csv('conflict.csv', index=False)
    results.to_json('conflict.json', orient='records', lines=True)
    print(results.head())
    df['prompt'] = df.apply(generate_normal, axis=1)
    df['true_label'] = df['correct_option']
    new_dataset = df[[ 'prompt','true_label']]

    # 保存为 CSV 文件
    new_dataset.to_csv('normal_dataset.csv', index=False)

    # 保存为 JSON 文件
    new_dataset.to_json('normal_dataset.json', orient='records', lines=True)


    print("生成的新数据集已保存为 CSV 和 JSON 文件。")
    print(new_dataset.head())