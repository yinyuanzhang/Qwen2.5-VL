from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration # <--- 这里修改了导入
import torch

# 模型在 Hugging Face Hub 上的名称
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

print(f"开始下载和加载模型: {model_id}")

try:
    # 加载处理器 (tokenizer 和 image processor)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("处理器下载并加载成功！")

    # 加载模型 - 这里是关键修改！
    # 使用 Qwen2_5_VLForConditionalGeneration 类来加载多模态模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained( # <--- 从这里修改
        model_id,
        torch_dtype=torch.bfloat16, # 如果你的GPU不支持bf16，可以改为torch.float16
        low_cpu_mem_usage=True,
        trust_remote_code=True # <--- 这个非常重要！
    ).cuda() # 将模型加载到 GPU

    print("模型下载并加载成功！")
    print(f"模型已加载到: {model.device}")

except Exception as e:
    print(f"下载或加载模型时发生错误: {e}")
    print("请检查你的网络连接，并确保transformers库已更新到最新版本，同时模型加载类正确。")