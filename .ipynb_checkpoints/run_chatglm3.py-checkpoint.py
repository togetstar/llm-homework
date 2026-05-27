from transformers import AutoTokenizer, AutoModel
import torch

model_dir = "/mnt/workspace/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_dir,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model.eval()

print("ChatGLM3-6B ready (type exit to quit)")

history = []

while True:
    text = input("User: ")
    if text.lower() in ["exit", "quit"]:
        break

    response, history = model.chat(
        tokenizer,
        text,
        history=history
    )

    print("Assistant:", response)