from itertools import combinations
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations


from pathlib import Path
# 映射表，将每个 Subject 映射到 BroaderSubject
SUBJECT_TO_BROADER_SUBJECT = {
    'astronomy_test': 'Nature Science',
    'college_biology_test': 'Nature Science',
    'college_chemistry_test': 'Nature Science',
    'conceptual_physics_test': 'Nature Science',
    'high_school_physics_test': 'Nature Science',
    'high_school_biology_test': 'Nature Science',
    'high_school_chemistry_test': 'Nature Science',
    'high_school_government_politics': 'Social Science',
    'high_school_macroeconomics_test': 'Social Science',
    'high_school_microeconomics_test': 'Social Science',
    'management_test': 'Social Science',
    'professional_accounting_test': 'Social Science',
    'sociology_test': 'Social Science',
    'us_foreign_policy_test': 'Social Science',
    'world_religions_test': 'Social Science',
    'high_school_psychology_test': 'Social Science',
    'electrical_engineering_test': 'Engineering',
    'college_computer_science_test': 'Engineering',
    'clinical_knowledge_test': 'Medicine',
    'college_medicine_test': 'Medicine',
    'medical_genetics_test': 'Medicine',
    'nutrition_test': 'Medicine',
    'virology_test': 'Medicine',
    'anatomy_test': 'Medicine',
    'global_facts_test': 'Humanities',
    'moral_disputes_test': 'Humanities',
    'miscellaneous_test': 'Humanities',
    'high_school_geography_test': 'Others',
    'logical_fallacies_test': 'Others',
    'human_aging_test': 'Others'
}
def calculate_accuracy_by_subject(json_file):
    """
    从单个模型的 JSONL 文件中按类别计算准确率。
    :param json_file: JSONL 文件路径，包含每个记录的类别和正确性
    :return: 每个类别的正确率字典
    """
    subject_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    # 逐行读取 JSONL 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())  # 解析每一行的 JSON 数据
            subject = entry["subject"]
            is_correct = entry["is_correct"]
            
            # 更新总数和正确数
            subject_counts[subject]["total"] += 1
            if is_correct:
                subject_counts[subject]["correct"] += 1

    # 计算每个类别的正确率
    subject_accuracy = {subject: (counts["correct"] / counts["total"]) * 100
                        for subject, counts in subject_counts.items()}
    
    return subject_accuracy
def calculate_accuracy_by_broader_subject(file_path):
    """
    从一个 JSONL 文件中读取正确预测数据，并返回按 Broader Subject 分组的准确率。
    :param file_path: JSONL 文件路径
    :return: 按 Broader Subject 分组的准确率
    """
    broader_subject_correct = defaultdict(int)  # 记录每个 Broader Subject 的正确预测数
    broader_subject_total = defaultdict(int)    # 记录每个 Broader Subject 的总预测数
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())  # 解析每一行的 JSON 数据
            subject = entry["subject"]
            is_correct = entry["is_correct"]
            
            # 根据映射表获取对应的 Broader Subject
            broader_subject = SUBJECT_TO_BROADER_SUBJECT.get(subject)
            # print(broader_subject)
            
            if broader_subject is not None and is_correct is not None:
                broader_subject_total[broader_subject] += 1  # 总数加一
                if is_correct:  # 如果预测正确
                    broader_subject_correct[broader_subject] += 1
    
    # 计算每个 Broader Subject 的准确率
    broader_subject_accuracies = {
        subject: (broader_subject_correct[subject] / broader_subject_total[subject]) * 100
        for subject in broader_subject_total.keys()
    }
    
    return broader_subject_accuracies
def gather_accuracies_from_models(model_dirs):
    """
    从多个模型文件中收集每个模型的准确率数据。
    :param model_files: 包含模型 JSON 文件路径的列表
    :return: 模型的准确率字典，每个模型的名称对应一个按类别计算的准确率字典
    """
    all_accuracies = {}
    for model_dir in model_dirs:
            if Path(model_dir).exists():
                for filename in os.listdir(model_dir):
                    file_path = os.path.join(model_dir, filename)
                    # print(file_path)

                    # and filename.split("_")[1] == "5shots"
                    # if file_path.endswith("jsonl"):
                    if file_path.endswith("jsonl") and filename.split("_")[1] != "5shots":
                        model_name = os.path.basename(model_dir)+filename.split("_")[1]
                        print(file_path)
                        accuracy_data = calculate_accuracy_by_broader_subject(file_path)
                        all_accuracies[model_name] = accuracy_data
    print(all_accuracies)
    return all_accuracies

def plot_spider_chart_for_models(accuracies_data, output_path="spider_chart.png"):
    """
    绘制多个模型的蜘蛛网图，并将图像保存到文件。
    :param accuracies_data: 字典，每个模型的名称对应按类别计算的正确率数据
    :param output_path: 图像保存路径
    """
    # 获取所有类别标签
    subjects = set()
    for accuracy in accuracies_data.values():
        subjects.update(accuracy.keys())
    subjects = sorted(subjects)  # 排序方便视觉对比

    # 初始化数据
    num_vars = len(subjects)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合环形图

    # 设置图的比例大小
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  # 缩小图的大小

    # 使用新的颜色映射方式
    colors = plt.colormaps["tab10"]  # 使用颜色映射
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']  # 标记样式列表

    # 绘制每个模型的数据
    for i, (model_name, accuracy_data) in enumerate(accuracies_data.items()):
        # 将准确率数据转换为列表，按顺序排列，缺失的类别用0补全
        data = [accuracy_data.get(subject, 0) for subject in subjects]
        data += data[:1]  # 闭合数据环形图

        # 使用不同颜色和标记绘制填充区域和外边框
        color = colors(i)
        ax.fill(angles, data, color=color, alpha=0.3, label=model_name)
        ax.plot(angles, data, color=color, linewidth=1.5, linestyle='-', marker=markers[i % len(markers)], markersize=5)

    # 设置类别标签和字体大小
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subjects, fontsize=8)  # 调整字体大小

    # 设置图例和比例
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_ylim(0, 30)  # 设置比例范围

    # 保存图像
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    print(f"蜘蛛网图已保存至 {output_path}")


import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from collections import defaultdict

def polar_to_cartesian(r, angles):
    """
    将极坐标（r, angle）转换为笛卡尔坐标（x, y）。
    :param r: 半径列表
    :param angles: 角度列表
    :return: x 和 y 坐标列表
    """
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return x, y

def calculate_overlap_area(accuracies_data):
    """
    计算多个模型之间的重叠面积。
    :param accuracies_data: 字典，每个模型的名称对应按类别计算的正确率数据
    :return: 重叠面积
    """
    # 获取所有类别标签，确保每个模型的数据结构一致
    subjects = set()
    for accuracy in accuracies_data.values():
        subjects.update(accuracy.keys())
    subjects = sorted(subjects)  # 排序方便视觉对比

    # 初始化数据
    num_vars = len(subjects)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合环形图

    # 创建模型多边形
    polygons = []
    for model_name, accuracy_data in accuracies_data.items():
        # 将准确率数据转换为列表，按顺序排列，缺失的类别用0补全
        data = [accuracy_data.get(subject, 0) for subject in subjects]
        data += data[:1]  # 闭合数据环形图

        # 转换为笛卡尔坐标
        x, y = polar_to_cartesian(np.array(data), np.array(angles))
        coordinates = list(zip(x, y))
        
        # 创建模型的多边形并检查有效性
        polygon = Polygon(coordinates)
        if polygon.is_valid and polygon.area > 0:  # 过滤掉无效或面积为零的多边形
            polygons.append(polygon)

    # 确保至少有两个有效多边形
    if len(polygons) < 2:
        print("没有足够的有效多边形进行重叠计算")
        return 0

    # 计算重叠面积
    overlap_area = polygons[0]
    for polygon in polygons[1:]:
        overlap_area = overlap_area.intersection(polygon)
        if overlap_area.is_empty:  # 如果重叠区域为空，直接返回 0
            return 0

    return overlap_area.area

def calculate_mae(model1_data, model2_data):
    """
    计算两个模型在多个类别上的平均绝对误差（MAE）。
    :param model1_data: 第一个模型的类别准确率数据
    :param model2_data: 第二个模型的类别准确率数据
    :return: MAE 值
    """
    all_categories = set(model1_data.keys()).union(set(model2_data.keys()))
    total_error = 0
    for category in all_categories:
        score1 = model1_data.get(category, 0)
        score2 = model2_data.get(category, 0)
        total_error += abs(score1 - score2)
    
    mae = total_error / len(all_categories)
    return mae

def generate_mae_matrix(accuracies_data):
    """
    生成模型之间 MAE 的混淆矩阵。
    :param accuracies_data: 包含所有模型的准确率数据字典
    :return: MAE 混淆矩阵和模型列表
    """
    model_names = list(accuracies_data.keys())
    num_models = len(model_names)
    
    # 初始化混淆矩阵
    mae_matrix = np.zeros((num_models, num_models))
    
    # 填充矩阵
    for i, j in combinations(range(num_models), 2):
        model1 = model_names[i]
        model2 = model_names[j]
        mae = calculate_mae(accuracies_data[model1], accuracies_data[model2])
        mae_matrix[i, j] = mae
        mae_matrix[j, i] = mae  # 对称矩阵
    
    return mae_matrix, model_names
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

def plot_mae_co_clustering(mae_matrix, model_names, output_path="./png/co_clustering_spider_chart.png"):
    """
    Perform Co-Clustering on the MAE matrix and plot the results.
    
    :param mae_matrix: MAE values matrix (2D numpy array)
    :param model_names: List of model names (used for annotating the axes)
    :param output_path: Path to save the output plot
    """
    # Create and fit the Spectral Coclustering model
    model = SpectralCoclustering(n_clusters=5, random_state=0)
    model.fit(mae_matrix)

    # Reorder the MAE matrix based on the clustering
    clustered_mae_matrix = mae_matrix[np.argsort(model.row_labels_)]
    clustered_mae_matrix = clustered_mae_matrix[:, np.argsort(model.column_labels_)]

    # Plot the original MAE matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(mae_matrix, xticklabels=model_names, yticklabels=model_names, annot=True, fmt=".2f",
                cmap="coolwarm", annot_kws={"size": 8})  # Adjust annotation font size
    plt.title("Original MAE Matrix", fontsize=12)
    plt.xlabel("Model", fontsize=10)
    plt.ylabel("Model", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Reduce font size for x-axis
    plt.yticks(rotation=0, fontsize=8)              # Reduce font size for y-axis
    plt.tight_layout()
    plt.savefig(f"{output_path}_original.png")
    plt.show()

    # Plot the clustered (co-clustered) MAE matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(clustered_mae_matrix, xticklabels=model_names, yticklabels=model_names, annot=True, fmt=".2f",
                cmap="coolwarm", annot_kws={"size": 8})  # Adjust annotation font size
    plt.title("After Co-Clustering (Reordered MAE Matrix)", fontsize=12)
    plt.xlabel("Model", fontsize=10)
    plt.ylabel("Model", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Reduce font size for x-axis
    plt.yticks(rotation=0, fontsize=8)              # Reduce font size for y-axis
    plt.tight_layout()
    plt.savefig(f"{output_path}_co_clustering.png")
    plt.show()

    # Print the consensus score
    consensus = consensus_score(model.biclusters_, (np.argsort(model.row_labels_), np.argsort(model.column_labels_)))
    print(f"Consensus score: {consensus:.3f}")
def plot_mae_matrix(mae_matrix, model_names,output_path):
    """
    绘制模型之间 MAE 的混淆矩阵，并缩小字体。
    :param mae_matrix: MAE 值的混淆矩阵
    :param model_names: 模型名称列表
    """
    plt.figure(figsize=(12, 10))
    
    # 调整注释字体大小
    sns.heatmap(mae_matrix, xticklabels=model_names, yticklabels=model_names, annot=True, fmt=".2f", 
                cmap="coolwarm", annot_kws={"size": 8})  # 将注释字体设置为 8
    
    # 调整 x 和 y 轴标签的字体大小
    plt.xticks(rotation=45, ha='right', fontsize=8)  # 缩小 x 轴字体
    plt.yticks(rotation=0, fontsize=8)               # 缩小 y 轴字体
    
    plt.title("Mean Absolute Error (MAE) Between Models", fontsize=12)  # 调整标题字体大小
    plt.xlabel("Model", fontsize=10)
    plt.ylabel("Model", fontsize=10)
    plt.tight_layout()  # 调整布局
    print("mae_matrix已输出")
    plt.savefig(output_path)

# 示例调用
original_paths = [          
               
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B",
              "/mnt/data/yuliangyan/instruction_tuning_models/llama3-ft",

              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct",
              "/mnt/data/yuliangyan/instruction_tuning_models/llama3-instruct-ft",

              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B",
              "/mnt/data/yuliangyan/instruction_tuning_models/llama31-ft",

              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct",
              "/mnt/data/yuliangyan/instruction_tuning_models/llama31-instruct-ft",

              "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1",
              "/mnt/data/yuliangyan/instruction_tuning_models/mistral-ft",

              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base",
              "/mnt/data/yuliangyan/instruction_tuning_models/deepseek-ft",

              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat",
              "/mnt/data/yuliangyan/instruction_tuning_models/deepseek-chat-ft",

              "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct",
              "/mnt/data/yuliangyan/instruction_tuning_models/deepseek-math-instruct-ft",

              "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B",
              "/mnt/data/yuliangyan/instruction_tuning_models/qwen25-ft",

              "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct",
              "/mnt/data/yuliangyan/instruction_tuning_models/phi3-instruct-ft",
]
relative_paths = [f"./{os.path.basename(path)}" for path in original_paths]


accuracies_data = gather_accuracies_from_models(relative_paths)
# 示例调用
# 生成 MAE 混淆矩阵
mae_matrix, model_names = generate_mae_matrix(accuracies_data)
# 绘制混淆矩阵
plot_mae_co_clustering(mae_matrix, model_names)
from scipy.cluster.hierarchy import linkage, dendrogram


from scipy.spatial.distance import squareform
# 使用单位矩阵表示 E
# E = np.eye(similarity_matrix.shape[0])  # 创建单位矩阵

# 计算距离矩阵
distance_matrix = mae_matrix
# print(distance_matrix)
compressed_dist_matrix = squareform(distance_matrix)

# 计算层次聚类（linkage）
Z = linkage(compressed_dist_matrix, method='single')  # 单链法，可改为 'complete', 'average', 'ward'

# 绘制树状图
plt.figure(figsize=(16, 20))
dendrogram(
    Z,
    labels=model_names,  # 使用 similarity_matrix 的索引作为标签
    leaf_rotation=45,  # 标签旋转角度
    leaf_font_size=10  # 标签字体大小
)
plt.title("Cluster")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
# plt.xticks(rotation=135)
plt.savefig("./MAE_CLUSTER.png")

# plot_mae_matrix(mae_matrix, model_names,output_path="./png/0shot")
# for i in range(0, len(relative_paths), 2):
#     model_path = relative_paths[i]
#     tuning_path = relative_paths[i + 1]
    
#     # Create the output path based on the model names, e.g., Llama-3, Mistral-7B
#     output_path = f"./png/{model_path.split('/')[-1]}_spider_chart.png"
#     accuracies_data = gather_accuracies_from_models([model_path,tuning_path])

#     # Call the plot_spider_chart_for_models function
#     plot_spider_chart_for_models(accuracies_data, output_path=output_path)



# 输出每对模型的 MAE 结果
# for model_pair, mae in mae_results.items():
#     print(f"模型 {model_pair[0]} 和 {model_pair[1]} 的平均绝对误差 (MAE) 为: {mae}")
# 计算重叠面积
# overlap_area = calculate_overlap_area(accuracies_data)
# print(f"模型之间的重叠面积为: {overlap_area}")
# plot_spider_chart_for_models(accuracies_data, output_path="comparison_spider_chart.png")
