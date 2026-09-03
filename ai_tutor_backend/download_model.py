from modelscope import snapshot_download

print("开始从国内源下载 Embedding 模型...")
# 将模型下载到 D 盘的 AI_Models 文件夹
model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=r'D:\AI_Models')

print("="*50)
print(f"模型下载完成！")
print(f"请在代码中替换为以下本地路径：\n{model_dir}")
print("="*50)