from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "/mnt/workspace/Qwen-1_8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model.eval()

while True:
    text = input("User: ")
    if text.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128
        )

    print("Assistant:", tokenizer.decode(outputs[0], skip_special_tokens=True))