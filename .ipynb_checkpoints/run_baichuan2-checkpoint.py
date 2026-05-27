from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "/mnt/workspace/Baichuan2-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model.eval()

print("Baichuan2-7B ready (type exit to quit)")

history = []

while True:
    text = input("User: ")
    if text.lower() in ["exit", "quit"]:
        break

    prompt = ""
    for h in history:
        prompt += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
    prompt += f"User: {text}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Assistant:", response)

    history.append({"user": text, "assistant": response})