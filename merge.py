import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 设置路径 (替换为你自己的本地路径或 HuggingFace 仓库名)
base_model_path = "allenai/Llama-3.1-Tulu-3-8B" 
lora_path = "AGNDM/tulu_qasper_lora_final"
output_dir = "./merged_model"

print("正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16, # 推荐使用 float16 节省内存
    device_map="cpu",          # 如果显存不够，可以用 "cpu" 慢慢合
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("正在合并 LoRA 权重...")
# 加载 LoRA 权重
lora_model = PeftModel.from_pretrained(base_model, lora_path)

# 合并并卸载 (将 LoRA 矩阵乘加回原模型权重中)
merged_model = lora_model.merge_and_unload()

print("正在保存合并后的模型...")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("合并完成！")