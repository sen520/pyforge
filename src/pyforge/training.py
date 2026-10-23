"""单 GPU QLoRA：数据预检、训练、完整恢复和适配器推理。"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import time
from pathlib import Path

from .encoding import CausalCollator, encode_record
from .paths import MODEL_CACHE_DIR, PROJECT_ROOT
from .training_config import check_resume, load_config, load_splits


def configure_hub_runtime(offline):
    # PEFT 保存时也会查询基础模型配置，必须让它遵守同一离线设置。
    os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    temporary.replace(path)


def gpu_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查驱动及 PyTorch CUDA 安装")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前训练配置要求 BF16，请检查 GPU 与 CUDA 运行时")
    free, total = torch.cuda.mem_get_info()
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2)}


def resolve_snapshot(model_name, revision, offline=False):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        path = snapshot_download(model_name, revision=revision, cache_dir=str(MODEL_CACHE_DIR), local_files_only=True)
    except LocalEntryNotFoundError:
        if offline:
            raise ValueError("项目模型缓存中没有指定版本，请先下载模型或移除 --offline") from None
        path = snapshot_download(model_name, revision=revision, cache_dir=str(MODEL_CACHE_DIR),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    snapshot = Path(path)
    if not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        raise ValueError("模型快照无法解析为固定 commit SHA")
    return snapshot, snapshot.name


def load_base(model_name, revision, offline=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    gpu_check()
    snapshot, resolved = resolve_snapshot(model_name, revision, offline)
    print("加载模型快照：" + resolved, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
        device_map={"": torch.cuda.current_device()}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    # 适配器记录 Hub ID，而不是训练机器上的绝对缓存路径。
    model.name_or_path = model_name
    model.config._name_or_path = model_name
    model.config._commit_hash = resolved
    return model, tokenizer, resolved


def encode_splits(rows, tokenizer, max_length):
    encoded, counts = {}, {}
    for name in ("train", "validation"):
        values = [encode_record(row, tokenizer, max_length) for row in rows[name]]
        encoded[name] = [value for value in values if value is not None]
        counts[name] = {"source": len(values), "kept": len(encoded[name]), "filtered_long": values.count(None),
            "tokens": sum(len(value["input_ids"]) for value in encoded[name]),
            "supervised_tokens": sum(sum(label != -100 for label in value["labels"]) for value in encoded[name])}
        if not encoded[name]:
            raise ValueError(name + " 编码后为空，请检查长度限制")
    return encoded, counts


def code_revision():
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), *arguments], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def train(config_path, data_dir, output_dir, resume=None, offline=False, stop_after=None):
    config = load_config(config_path)
    rows, hashes = load_splits(data_dir)
    output = Path(output_dir).resolve()
    if stop_after is not None and (type(stop_after) is not int or stop_after <= 0):
        raise ValueError("stop-after 必须为正整数的全局更新步数")
    previous = state = None
    if resume:
        previous, state = check_resume(resume, output, config, hashes)
        if stop_after is not None and stop_after <= state["global_step"]:
            raise ValueError("stop-after 必须大于恢复检查点步数")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空；请使用新目录或通过 --resume 恢复完整检查点")

    configure_hub_runtime(offline)
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    hardware = gpu_check()
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}
    if previous and previous["versions"] != versions:
        raise ValueError("恢复环境的训练依赖版本与原运行不同")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".training.lock"
    try:
        lock_stream = lock.open("x")
    except FileExistsError:
        raise ValueError("训练锁已存在，请确认没有其他训练进程后再处理") from None
    lock_stream.close()
    began = time.monotonic()
    try:
        write_json(output / "run_status.json", {"status": "loading", "resume": str(resume) if resume else None})
        set_seed(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        revision = previous["model_revision"] if previous else config["revision"]
        model, tokenizer, revision = load_base(config["model"], revision, offline)
        encoded, counts = encode_splits(rows, tokenizer, config["max_length"])
        targets = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        missing = set(config["target_modules"]) - targets
        if missing:
            raise ValueError("基础模型缺少 LoRA 目标模块：" + ", ".join(sorted(missing)))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none", task_type="CAUSAL_LM"), revision=revision)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not trainable or any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
            raise RuntimeError("可训练参数不符合仅训练 LoRA 的要求")
        systems = {r["messages"][0]["content"] for r in rows["train"] if r["messages"][0]["role"] == "system"}
        manifest = {"schema_version": 1, "config": config, "model_revision": revision, "hardware": hardware,
            "versions": versions, "encoding": counts, "data_sha256": hashes, "code": code_revision(),
            "trainable_parameters": trainable, "system_prompt": next(iter(systems)) if len(systems) == 1 else None,
            "tokenizer": {"pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}}
        if previous:
            for key in ("model_revision", "tokenizer", "encoding"):
                if previous[key] != manifest[key]:
                    raise ValueError("恢复时模型或分词结果变化：" + key)
        else:
            write_json(output / "run_manifest.json", manifest)
        print(json.dumps({"gpu": hardware["gpu"], "trainable_parameters": trainable, "data": counts}, ensure_ascii=False), flush=True)

        class Progress(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                stopping = stop_after is not None and state.global_step >= stop_after
                if stopping:
                    control.should_training_stop = True
                if stopping or state.global_step >= state.max_steps:
                    control.should_save = True
                    control.should_evaluate = True
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                if any(key in logs and not math.isfinite(float(logs[key])) for key in ("loss", "eval_loss", "grad_norm")):
                    raise RuntimeError("训练产生非有限 loss 或梯度，请检查数据和精度")
                event = {"step": state.global_step, **logs}
                with (output / "training_log.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                write_json(output / "run_status.json", {"status": "training", "step": state.global_step, "max_steps": state.max_steps})

        args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"],
            num_train_epochs=config["epochs"], max_steps=config["max_steps"], bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, lr_scheduler_type="cosine", warmup_ratio=config["warmup_ratio"],
            weight_decay=config["weight_decay"], max_grad_norm=config["max_grad_norm"], logging_steps=1, logging_nan_inf_filter=False,
            eval_strategy="steps", eval_steps=config["eval_steps"], save_strategy="steps", save_steps=config["save_steps"],
            save_total_limit=config["save_total_limit"], report_to="none", seed=config["seed"], data_seed=config["seed"],
            dataloader_num_workers=0, optim="adamw_torch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, label_names=["labels"])
        trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
            data_collator=CausalCollator(tokenizer.pad_token_id), processing_class=tokenizer, callbacks=[Progress()])
        result = trainer.train(resume_from_checkpoint=str(Path(resume).resolve()) if resume else None)
        metrics = {**result.metrics, **trainer.evaluate(), "global_step": trainer.state.global_step,
            "planned_steps": trainer.state.max_steps, "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric, "trainable_parameters": trainable,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "elapsed_seconds": time.monotonic() - began}
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(str(output / "adapter"))
        trainer.save_state()
        metrics["elapsed_seconds"] = time.monotonic() - began
        write_json(output / "metrics.json", metrics)
        status = "completed" if trainer.state.global_step >= trainer.state.max_steps else "stopped"
        write_json(output / "run_status.json", {"status": status, "step": trainer.state.global_step,
            "max_steps": trainer.state.max_steps, "best_checkpoint": trainer.state.best_model_checkpoint})
        return metrics
    except BaseException as exc:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def chat(run_dir, prompt, max_new_tokens=512, offline=False):
    configure_hub_runtime(offline)
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    if not isinstance(prompt, str) or not prompt.strip() or max_new_tokens <= 0:
        raise ValueError("问题不能为空，max-new-tokens 必须为正数")
    path = Path(run_dir)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    model, _, _ = load_base(manifest["config"]["model"], manifest["model_revision"], offline)
    tokenizer = AutoTokenizer.from_pretrained(str(path / "adapter"), local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, str(path / "adapter"))
    model.eval()
    model.config.use_cache = True
    messages = ([{"role": "system", "content": manifest["system_prompt"]}] if manifest.get("system_prompt") else [])
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    if inputs.shape[1] > manifest["config"]["max_length"]:
        raise ValueError("输入超过训练配置长度，请缩短输入")
    if inputs.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("输入与生成长度超过模型上下文长度")
    with torch.inference_mode():
        generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
