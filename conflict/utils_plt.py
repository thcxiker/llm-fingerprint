import numpy as np
import matplotlib.pyplot as plt

def create_spider_chart(dataset1, dataset2, num_vars=360, label1="PGDC (Ours)", label2="Baseline P-dis"):
    """
    生成一个极坐标蛛网图，用于展示两个数据集在多个方向上的对比。

    参数:
    - dataset1: 第一组数据，列表或数组，每个值为一个方向的数值（范围0-100）
    - dataset2: 第二组数据，列表或数组，每个值为一个方向的数值（范围0-100）
    - num_vars: 数据方向的数量，默认为360
    - label1: 第一组数据的标签，默认为 "PGDC (Ours)"
    - label2: 第二组数据的标签，默认为 "Baseline P-dis"
    """
    # 确保输入数据的长度与方向数量一致
    assert len(dataset1) == num_vars, "dataset1 的长度必须等于 num_vars"
    assert len(dataset2) == num_vars, "dataset2 的长度必须等于 num_vars"
    
    # 生成角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 创建一个极坐标图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 绘制两组数据
    ax.fill(angles, dataset1, color="skyblue", alpha=0.5, label=label1)
    ax.fill(angles, dataset2, color="lightblue", alpha=0.5, label=label2)

    # 设置图例
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    # 添加比例标签
    ax.set_ylim(0, 100)  # 设置比例范围0到100
    plt.show()

# 示例调用
# 创建模拟数据（范围0-100）
num_vars = 360
dataset1 = np.random.rand(num_vars) * 100
dataset2 = np.random.rand(num_vars) * 100

# 调用函数生成图表
create_spider_chart(dataset1, dataset2, num_vars)
