import matplotlib.pyplot as plt
import numpy as np

# 1. 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 你的真实数据
control_scores = [60, 90, 30, 110, 140, 70, 120, 110, 140, 100]
experiment_scores = [130, 120, 130, 120, 110, 120, 120, 130, 120, 130]

# 3. 统计每个分数出现的次数（按实际分数点统计，和参考图逻辑一致）
unique_scores = sorted(list(set(control_scores + experiment_scores)))
control_counts = [control_scores.count(s) for s in unique_scores]
experiment_counts = [experiment_scores.count(s) for s in unique_scores]

# 4. 设置柱子宽度和位置
width = 0.35  # 单根柱子的宽度
x = np.arange(len(unique_scores))  # 每个分数点的位置

# 5. 创建画布并绘图
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, control_counts, width, 
        label='对照组 (普通视频)', color='#1f77b4')
plt.bar(x + width/2, experiment_counts, width, 
        label='实验组 (智能导学)', color='#ff7f0e')

# 6. 美化图表
plt.title('两组学生客观测试成绩分布对比', fontsize=14, pad=15)
plt.xlabel('测试分数', fontsize=12)
plt.ylabel('学生人数', fontsize=12)

# 设置X轴刻度为实际分数
plt.xticks(x, unique_scores)
plt.ylim(0, max(max(control_counts), max(experiment_counts)) + 1)  # 顶部留一点空间

# 添加水平网格线，让人数对比更清晰
plt.grid(axis='y', linestyle='--', alpha=0.4)

# 去除顶部和右侧边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.tight_layout()
plt.show()