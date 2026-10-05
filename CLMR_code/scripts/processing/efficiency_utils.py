
from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable

import torch


def first_tensor(batch):
    if torch.is_tensor(batch):
        return batch
    values = batch.values() if isinstance(batch, dict) else batch
    for item in values:
        candidate = first_tensor(item) if isinstance(item, (dict, tuple, list)) else (item if torch.is_tensor(item) else None)
        if candidate is not None:
            return candidate
    return None


def move_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    return batch


def collect_batches(loader: Iterable, count: int, device: torch.device) -> list:
    batches = []
    for batch in loader:

        tensor = first_tensor(batch)
        if tensor is not None and tensor.shape[0] != loader.batch_size:
            continue
        batches.append(move_to_device(batch, device))
        if len(batches) == count:
            break
    if len(batches) < count:
        raise RuntimeError(f"Need {count} full batches, found only {len(batches)}")
    return batches


@torch.inference_mode()
def benchmark_cuda(
    model: torch.nn.Module,
    batches: list,
    forward_fn: Callable,
    *,
    warmup_batches: int = 10,
    repeats: int = 3,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal efficiency benchmark")
    device = next(model.parameters()).device
    model.eval()

    def run(batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            forward_fn(model, batch)

    for batch in batches[:warmup_batches]:
        run(batch)
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model_memory = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        instances = 0
        for batch in batches:
            run(batch)
            instances += int(first_tensor(batch).shape[0])
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device)
        samples.append({
            "seconds": seconds,
            "instances_per_second": instances / seconds,
            "milliseconds_per_instance": 1000.0 * seconds / instances,
            "peak_gpu_gb": peak / (1024 ** 3),
            "incremental_peak_gpu_gb": (peak - model_memory) / (1024 ** 3),
        })

    def summarize(key: str) -> tuple[float, float]:
        values = [sample[key] for sample in samples]
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    throughput, throughput_std = summarize("instances_per_second")
    latency, latency_std = summarize("milliseconds_per_instance")
    peak, peak_std = summarize("peak_gpu_gb")
    incremental, incremental_std = summarize("incremental_peak_gpu_gb")
    return {
        "instances": len(batches) * int(first_tensor(batches[0]).shape[0]),
        "batches": len(batches),
        "repeats": repeats,
        "instances_per_second_mean": throughput,
        "instances_per_second_std": throughput_std,
        "milliseconds_per_instance_mean": latency,
        "milliseconds_per_instance_std": latency_std,
        "peak_gpu_gb_mean": peak,
        "peak_gpu_gb_std": peak_std,
        "incremental_peak_gpu_gb_mean": incremental,
        "incremental_peak_gpu_gb_std": incremental_std,
        "raw_repeats": samples,
    }


def model_size(model: torch.nn.Module) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
