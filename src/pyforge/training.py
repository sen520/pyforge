"""单 GPU QLoRA 训练与适配器推理。重型依赖仅在调用时导入。"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from .data import read_records
from .encoding import CausalCollator, encode_record
from .paths import MODEL_CACHE_DIR


def gpu_check():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查驱动及 PyTorch CUDA 安装")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前环境不支持 BF16；请使用经过验证的 GPU 与运行时")
    free, total = torch.cuda.mem_get_info()
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2)}


def load_base(model_name, revision):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    gpu_check()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, cache_dir=str(MODEL_CACHE_DIR), trust_remote_code=False, device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    resolved = getattr(model.config, "_commit_hash", None)
    if not resolved:
        raise RuntimeError("无法解析基础模型 commit SHA，拒绝启动不可追溯的运行")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved,
                                             cache_dir=str(MODEL_CACHE_DIR), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, resolved


def train(config_path, data_dir, output_dir, resume=None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    positive = ("max_length", "lora_rank", "lora_alpha", "gradient_accumulation_steps", "learning_rate", "epochs")
    for key in positive:
        if not isinstance(config.get(key), (int, float)) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError("配置必须为正数：" + key)
    for key in ("max_length", "lora_rank", "lora_alpha", "gradient_accumulation_steps", "seed", "max_steps"):
        if type(config.get(key)) is not int:
            raise ValueError("配置必须为整数：" + key)
    if config["max_steps"] != -1 and config["max_steps"] <= 0:
        raise ValueError("max_steps 必须为 -1 或正整数")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("输出目录非空；请使用新目录或显式指定 --resume")
    if resume and not (Path(resume) / "trainer_state.json").is_file():
        raise ValueError("--resume 必须指向完整 Trainer 检查点")
    rows = {name: read_records(Path(data_dir) / (name + ".jsonl")) for name in ("train", "validation", "test")}
    for field in ("id", "group_id"):
        sets = [{row[field] for row in rows[name]} for name in rows]
        if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("跨集合 {} 泄漏".format(field))
    conversations = [{json.dumps(row["messages"], sort_keys=True, ensure_ascii=False) for row in values}
                     for values in rows.values()]
    if any(conversations[a] & conversations[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("跨集合重复对话泄漏")
    set_seed(config["seed"])
    hardware = gpu_check()
    model, tokenizer, revision = load_base(config["model"], config["revision"])
    encoded, counts = {}, {}
    for name in ("train", "validation"):
        values = [encode_record(row, tokenizer, config["max_length"]) for row in rows[name]]
        encoded[name] = [value for value in values if value is not None]
        counts[name] = {"kept": len(encoded[name]), "filtered_long": values.count(None)}
        if not encoded[name]:
            raise ValueError("{} 编码后为空，请检查长度限制".format(name))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("没有可训练参数")
    if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
        raise RuntimeError("发现非 LoRA 可训练权重")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "model_revision": revision, "hardware": hardware, "encoding": counts,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
        "data_sha256": {name: hashlib.sha256((Path(data_dir) / (name + ".jsonl")).read_bytes()).hexdigest() for name in rows}}
    manifest_path = output / "运行清单.json"
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("config", "model_revision", "data_sha256"):
            if previous[key] != manifest[key]:
                raise ValueError("恢复时运行清单不匹配：" + key)
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"], max_steps=config["max_steps"], bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, max_grad_norm=1.0, logging_steps=1, eval_strategy="steps", eval_steps=100,
        save_steps=100, save_total_limit=2, report_to="none", seed=config["seed"], dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
                      data_collator=CausalCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = {**result.metrics, **trainer.evaluate(), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    trainer.save_model(str(output / "适配器"))
    tokenizer.save_pretrained(str(output / "适配器"))
    # 保存完整结束检查点，以支持不足 100 步的冒烟运行恢复。
    trainer._save_checkpoint(trainer.model, trial=None)
    trainer.save_state()
    (output / "训练指标.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def chat(run_dir, prompt, max_new_tokens=512):
    import torch
    from peft import PeftModel

    path = Path(run_dir)
    manifest = json.loads((path / "运行清单.json").read_text(encoding="utf-8"))
    model, tokenizer, _ = load_base(manifest["config"]["model"], manifest["model_revision"])
    model = PeftModel.from_pretrained(model, str(path / "适配器"))
    model.eval()
    model.config.use_cache = True
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt").to(model.device)
    if inputs.shape[1] > manifest["config"]["max_length"]:
        raise ValueError("输入超过训练配置长度，请缩短输入")
    with torch.inference_mode():
        generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
