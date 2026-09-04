"""Benchmark one real optimizer update of the target KitAI architecture on CUDA."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.config import ModelConfig
from models.transformer import KitAITransformer
from training.optimizer import create_optimizer
from utils.io_utils import read_yaml


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def host_memory() -> dict[str, int]:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {"total_bytes": int(status.ullTotalPhys), "available_bytes": int(status.ullAvailPhys)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the target-hardware benchmark")
    full_config = read_yaml(args.config)
    model_config = ModelConfig(**full_config["model"])
    train_config = full_config["training"]
    batch_size = int(train_config["batch_size"])
    grad_accumulation = int(train_config["gradient_accumulation_steps"])
    seq_len = int(model_config.max_seq_len)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    host_before = host_memory()

    model = KitAITransformer(model_config).to(device).train()
    optimizer = create_optimizer(
        model,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=(0.9, 0.95),
        fused=False,
    )
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    inputs = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)

    # A small warm-up builds CUDA kernels before the timed full accumulation cycle.
    with torch.amp.autocast("cuda", enabled=True):
        warmup_loss, _ = model.forward_with_loss(inputs, labels)
        warmup_loss = warmup_loss / grad_accumulation
    scaler.scale(warmup_loss).backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(grad_accumulation):
        with torch.amp.autocast("cuda", enabled=True):
            loss, _ = model.forward_with_loss(inputs, labels)
            loss = loss / grad_accumulation
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    host_after = host_memory()
    tokens = batch_size * seq_len * grad_accumulation
    report = {
        "passed": True,
        "device": torch.cuda.get_device_name(0),
        "model_trainable_parameters": model.get_trainable_parameters(),
        "microbatch_size": batch_size,
        "gradient_accumulation_steps": grad_accumulation,
        "effective_batch_size": batch_size * grad_accumulation,
        "sequence_length": seq_len,
        "tokens_per_optimizer_step": tokens,
        "elapsed_seconds_per_optimizer_step": elapsed,
        "measured_tokens_per_second": tokens / elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "host_memory_before": host_before,
        "host_memory_after": host_after,
        "host_available_ram_delta_bytes": host_before["available_bytes"] - host_after["available_bytes"],
        "gradient_norm": float(grad_norm),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
