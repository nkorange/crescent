import urllib.request
import zipfile
import os
import tiktoken
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
from 


url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
	if data_file_path.exists():
		print(f"{data_file_path} already exists. Skipping download and extraction.")
		return
	with urllib.request.urlopen(url) as response:
		with open(zip_path, "wb") as out_file:
			out_file.write(response.read())

	with zipfile.ZipFile(zip_path, "r") as zip_ref:
		zip_ref.extractall(extracted_path)

	original_file_path = Path(extracted_path) / "SMSSpamCollection"
	os.rename(original_file_path, data_file_path)
	print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
	num_spam = df[df["Label"] == "spam"].shape[0]
	ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
	balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
	return balanced_df

def random_split(df, train_frac, validation_frac):
	df = df.sample(frac=1, random_state=123).reset_index(drop=True)
	train_end = int(len(df) * train_frac)
	validation_end = train_end + int(len(df) * validation_frac)

	train_df = df[:train_end]
	validation_df = df[train_end:validation_end]
	test_df = df[validation_end:]

	return train_df, validation_df, test_df

def generate_text_simple(model, idx, max_new_tokens, context_size):
	for _ in range(max_new_tokens):
		idx_cond = idx[:, -context_size:]
		with torch.no_grad():
			logits = model(idx_cond)

		logits = logits[:, -1, :]
		probas = torch.softmax(logits, dim=-1)
		idx_next = torch.argmax(probas, dim=-1, keepdim=True)
		idx = torch.cat((idx, idx_next), dim=1)

	return idx

def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
	encoded_tensor = torch.tensor(encoded).unsqueeze(0)
	return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0)
	return tokenizer.decode(flat.tolist())

def assign(left, right):
	if left.shape != right.shape:
		raise ValueError(f"Shape mismatch. Left: {Left.shape}, Right: {Right.shape}")
	return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
	gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
	gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

	for b in range(len(params["block"])):
		q_w, k_w, v_w = np.split((params["block"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
		gpt.trf_blocks.att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
		gpt.trf_blocks.att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
		gpt.trf_blocks.att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

		q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
		gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
		gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
		gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

		gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
		gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

		gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["b"])
		gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["block"][b]["mlp"]["c_fc"]["b"])
		gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["b"])
		gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["block"][b]["mlp"]["c_proj"]["b"])

		gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
		gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
		gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
		gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

	gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
	gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
	gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])



class SpamDataset(Dataset):
	def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
		self.data = pd.read_csv(csv_file)
		self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
		if max_length is None:
			self.max_length = self._longest_encoded_length()
		else:
			self.max_length = max_length
			self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]

		self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

	def __getitem__(self, index):
		encoded = self.encoded_texts[index]
		label = self.data.iloc[index]["Label"]
		return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))

	def __len__(self):
		return len(self.data)

	def _longest_encoded_length(self):
		max_length = 0
		for encoded_text in self.encoded_texts:
			encoded_length = len(encoded_text)
			if encoded_length > max_length:
				max_length = encoded_length
		return max_length 


class GPTModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
		self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
		self.drop_emb = nn.Dropout(cfg["drop_rate"])

		self.trf_blocks = nn.Sequential(
			*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

		self.final_norm = LayerNorm(cfg["emb_dim"])
		self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

	def forward(self, in_idx):
		batch_size, seq_len = in_idx.shape
		tok_embeds = self.tok_emb(in_idx)

		pos_embeds = self.pos_emb(
			torch.arange(seq_len, device=in_idx.device)
		)
		x = tok_embeds + pos_embeds
		x = self.drop_emb(x)
		x = self.trf_blocks(x)
		x = self.final_norm(x)
		logits = self.out_head(x)
		return logits


download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df["Label"].value_counts())

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer = tokenizer)
val_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer = tokenizer)
test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer = tokenizer)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

for input_batch, target_batch in train_loader:
	pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions:", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
	"vocab_size": 50257,
	"context_length": 1024,
	"drop_rate": 0.0,
	"qkv_bias": True
}
model_configs = {
	"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_headers": 12},
	"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_headers": 16},
	"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_headers": 20},
	"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_headers": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text_1 = "Every effort moves you"
token_ids = generate_text_simple(model=model, idx=text_to_token_ids(text_1, tokenizer), max_new_tokens=15, context_size=BASE_CONFIG["context_length"])
print(token_ids_to_text(token_ids, tokenizer))


















