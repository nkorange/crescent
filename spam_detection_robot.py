from spam import *


torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(BASE_CONFIG)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
model.to(device)
model_state_dict = torch.load("spam_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)
model.eval()

# text_1 = (
#     "You are a winner you have been specially"
#     " selected to receive $1000 cash or a $2000 award."
# )
#
# print(classify_review(
#     text_1, model, tokenizer, device, max_length=120
# ))
#
# text_2 = (
#     "Hey, just wanted to check if we're still on"
#     " for dinner tonight? Let me know!"
# )
#
# print(classify_review(
#     text_2, model, tokenizer, device, max_length=120
# ))

print("Hi, I'm a robot to detect SPAM texts, please the text you want to detect if it's SPAM. Input exit to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['退出', 'exit', 'quit']:
        print("Robot: Bye!")
        break
    else:
        print("Robot: Your input is →", user_input)
        ans = classify_review(user_input, model, tokenizer, device, max_length=120)
        print("Robot: The result is", ans)