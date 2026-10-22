"""用模板前缀校验构造 assistant-only 标签；不匹配时拒绝训练。"""


def encode_record(record, tokenizer, max_length):
    messages = record["messages"]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(tokens) > max_length:
        return None
    labels = [-100] * len(tokens)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, add_generation_prompt=False)
        if tokens[:len(prefix)] != prefix or tokens[:len(through)] != through or len(through) <= len(prefix):
            raise ValueError("chat template 不支持可验证的前缀标签屏蔽")
        labels[len(prefix):len(through)] = tokens[len(prefix):len(through)]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("样本没有有效的 assistant 监督 token")
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": labels}


class CausalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        length = max(len(row["input_ids"]) for row in features)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in features:
            padding = length - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
