from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# 確保模型在 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# 創建一個簡單的輸入
tokenizer.src_lang = "en"
input_text = "Hello, world!"

encoded_input = tokenizer(input_text, return_tensors="pt").to(device)  # 確保輸入也在 GPU 上




# 嘗試運行模型
with torch.no_grad():
    output = model.generate(encoded_input["input_ids"])
# 嘗試運行模型