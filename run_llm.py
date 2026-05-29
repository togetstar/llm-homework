import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM
)

print("=" * 50)
print("大模型对话系统")
print("1. Qwen-1.8B-Chat")
print("2. ChatGLM3-6B")
print("=" * 50)

choice = input("请选择模型(1/2): ")

# =====================================================
# Qwen
# =====================================================

if choice == "1":

    model_path = "/mnt/workspace/Qwen-1_8B-Chat"

    print("\n正在加载 Qwen 模型，请稍等...\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).eval()

    print("Qwen 加载成功！")
    print("输入 exit 退出\n")

    while True:

        prompt = input("用户: ")

        if prompt.lower() == "exit":
            break

        print("\nQwen 正在思考...\n")

        response, history = model.chat(
            tokenizer,
            prompt,
            history=None
        )

        print("Qwen:")
        print(response)
        print()


# =====================================================
# ChatGLM3
# =====================================================

elif choice == "2":

    model_path = "/mnt/workspace/chatglm3-6b"

    print("\n正在加载 ChatGLM3 模型，请稍等...\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).eval()

    print("ChatGLM3 加载成功！")
    print("输入 exit 退出\n")

    history = []

    while True:

        prompt = input("用户: ")

        if prompt.lower() == "exit":
            break

        print("\nChatGLM3 正在思考...\n")

        response, history = model.chat(
            tokenizer,
            prompt,
            history=history
        )

        print("ChatGLM3:")
        print(response)
        print()


# =====================================================
# 输入错误
# =====================================================

else:

    print("输入错误，请重新运行程序！")