import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, CLIPProcessor
import onnxruntime as ort
import matplotlib

from transformers import AutoImageProcessor, AutoTokenizer
matplotlib.rc("font", family='Microsoft YaHei')
def create_onnx_session(model_path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用GPU
    return ort.InferenceSession(model_path, providers=providers)
# 加载原始CLIP模型用于文本编码
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True, cache_dir="./")
image_processor = AutoImageProcessor.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True, cache_dir="./")
# session = create_onnx_session('model_q4f16.onnx')  
model=AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True, cache_dir="./")
model.to('cuda')
# 初始化ONNX图像编码会话
def create_onnx_session(model_path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用GPU
    return ort.InferenceSession(model_path, providers=providers)

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from pathlib import Path

def create_onnx_session(model_path):
    """创建ONNX推理会话"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用GPU
    return ort.InferenceSession(model_path, providers=providers)

# 图像编码函数
def encode_images(image_paths, batch_size=16):
    """对图像批量编码为特征向量"""
    valid_paths = []
    pixel_values=[]
    for i in tqdm(range(0, len(image_paths), batch_size), desc="处理图像"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_valid_paths = []

        for img_path in batch_paths:
            try:
                print(f"Attempting to open {img_path}")
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                batch_valid_paths.append(img_path)
                print(f"Successfully load {img_path}")
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue

        if not batch_images:
            continue

        # 使用image_processor处理图像
        # pixel_values = image_processor(batch_images, return_tensors="np")['pixel_values']
        pixel_values.append(model.encode_image(batch_images))
        # 使用ONNX模型进行图像编码
        valid_paths.extend(batch_valid_paths)
    return np.concatenate(pixel_values, axis=0), valid_paths
# 文本编码函数
def encode_text(text_queries):
    """将文本查询编码为特征向量"""
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    
    # 使用tokenizer处理文本
    # tokenized = tokenizer(text_queries, return_tensors="np", padding=True)
    # input_ids = tokenized['input_ids'].astype(np.float32)
    input_ids=model.encode_text(  text_queries, task='retrieval.query')
    

    
    return input_ids

# 相似度计算和排序
def search_images(text_embedding, pixel_values, image_paths, top_k=5):
    """根据文本嵌入搜索最相似的图像"""
    similarities = []
    for pixel_value in pixel_values:
        # 确保image_embedding的形状是(1, embedding_dim)
        # _, _, text_embeddings, image_embeddings = session.run(None, {'input_ids': text_embedding.astype(np.int64), 'pixel_values': pixel_value.unsqueeze(0).numpy()})#假设输出是一个包含相似度分数的列表
        # similarity=text_embeddings @ image_embeddings[0].T
        similarity=str(text_embedding @ pixel_value.T)

        similarities.append(similarity)

    similarities = np.array(similarities)
    top_indices = np.argsort(similarities.flatten())[::-1][:top_k]
    top_similarities = similarities[top_indices]
    top_images = [image_paths[i] for i in top_indices]

    return list(zip(top_images, top_similarities))

# 展示结果函数
def display_results(results, query):
    """可视化展示搜索结果"""
    plt.figure(figsize=(15, 5 * ((len(results) + 2) // 3)))
    plt.suptitle(f'result: "{query}"', fontsize=16)
    
    for i, (img_path, similarity) in enumerate(results):
        plt.subplot(((len(results) + 2) // 3), 3, i + 1)
        try:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"similarity: {similarity:.4f}\n{Path(img_path).name}")
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"无法加载图像: {e}", ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# 主函数：扫描图像、建立索引并实现搜索
def main():
    # 设置图像文件夹路径
    # image_folder = input("请输入图像文件夹路径: ")
    image_folder = r'C:\Users\H\Pictures\test'
    
    # 支持的图像扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    
    # 获取所有图像文件路径
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, "**", ext), recursive=True))
    
    print(f"找到 {len(image_paths)} 个图像文件")
    
    if not image_paths:
        print("未找到图像，请检查文件夹路径是否正确")
        return
    
    # 检查是否存在预先计算的索引
    index_path = os.path.join(image_folder, "clip_image_index.npz")
    if os.path.exists(index_path):
        print(f"加载已有索引: {index_path}")
        data = np.load(index_path, allow_pickle=True)
        image_embeddings = data['embeddings']
        stored_paths = data['paths'].tolist()
        
        # 检查索引中的图像是否仍然存在
        valid_indices = []
        valid_paths = []
        for i, path in enumerate(stored_paths):
            if os.path.exists(path):
                valid_indices.append(i)
                valid_paths.append(path)
        
        image_embeddings = image_embeddings[valid_indices]
        image_paths = valid_paths
        print(f"从索引加载了 {len(image_paths)} 个有效图像")
    else:
        # 编码所有图像
        print("正在编码图像...")
        image_embeddings, image_paths = encode_images(image_paths)
        
        # 保存索引以便将来使用
        # np.savez(index_path, 
        #         embeddings=image_embeddings, 
        #         paths=np.array(image_paths, dtype=object))
        # print(f"索引已保存到 {index_path}")
    
    # 开始搜索循环
    while True:
        query = input("\n请输入搜索关键词 (输入'q'退出): ")
        if query.lower() == 'q':
            break

        top_k = int(input("显示多少个结果? ") or "5")

        # 编码查询文本
        print("处理查询...")
        text_embedding = encode_text(query)
        # print(image_embeddings.shape)
        # 搜索图像
        results = search_images(text_embedding, image_embeddings, image_paths, top_k)

        # 显示结果
        display_results(results, query)

if __name__ == "__main__":
    main()
