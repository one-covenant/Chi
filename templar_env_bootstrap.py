#!/usr/bin/env python3
"""
Bootstrap an affinetes-compatible environment for TPS evaluation.

Creates an env folder containing:
  - env.py (Actor.evaluate for TPS)
  - requirements.txt
  - Dockerfile
  - train.py (copied from templar-tournament or user-provided path)

This env is compatible with validator.py's expected Actor.evaluate signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ENV_PY = """\
import ast
import hashlib
import os
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

# Miner submission (must define inner_steps + InnerStepsResult)
import train

OUTPUT_VECTOR_TOLERANCE = float(os.getenv("OUTPUT_VECTOR_TOLERANCE", "0.02"))
VERIFY_LOSS = os.getenv("VERIFY_LOSS", "0") == "1"
LOSS_TOLERANCE = float(os.getenv("LOSS_TOLERANCE", "1e-3"))
DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", "1") == "1"
EVAL_SEQUENCE_LENGTH = int(os.getenv("EVAL_SEQUENCE_LENGTH", "1024"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/templar_env"))

# Global cache for model/data across evaluations
_CACHE = {
    "model": None,
    "model_dir": None,
    "data": None,
    "data_path": None,
    "initial_state": None,
}


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def _download(url: str, dst: Path) -> None:
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def _maybe_extract(archive_path: Path, extract_dir: Path) -> Path:
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        return archive_path

    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_dir


def _set_deterministic(seed: int) -> None:
    if not DETERMINISTIC_MODE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_code_structure(code_path: Path) -> tuple[bool, str | None]:
    try:
        code = code_path.read_text()
    except Exception as exc:
        return False, f"Failed to read train.py: {exc}"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}"

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                return False, (
                    f"inner_steps has {len(args.args)} args, expected at least 5"
                )
            break

    if not inner_steps_found:
        return False, "Missing required function: inner_steps"

    return True, None


def _validate_return_type(result) -> tuple[bool, str | None, InnerStepsResult | None]:
    if isinstance(result, InnerStepsResult):
        return True, None, result

    if all(hasattr(result, attr) for attr in ("final_logits", "total_tokens", "final_loss")):
        return True, None, InnerStepsResult(
            final_logits=result.final_logits,
            total_tokens=result.total_tokens,
            final_loss=result.final_loss,
        )

    return False, f"Invalid return type from inner_steps: {type(result)}", None


def _load_model(model_dir: Path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _load_data(data_path: Path):
    return torch.load(data_path, weights_only=True)


def _get_cached_model(model_dir: Path):
    cached = _CACHE.get("model")
    cached_dir = _CACHE.get("model_dir")
    if cached is not None and cached_dir == str(model_dir):
        return cached
    model = _load_model(model_dir)
    _CACHE["model"] = model
    _CACHE["model_dir"] = str(model_dir)
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model


def _get_cached_data(data_path: Path):
    cached = _CACHE.get("data")
    cached_path = _CACHE.get("data_path")
    if cached is not None and cached_path == str(data_path):
        return cached
    data = _load_data(data_path)
    _CACHE["data"] = data
    _CACHE["data_path"] = str(data_path)
    return data


def _create_data_iterator(data, batch_size: int, sequence_length: int) -> Iterator[torch.Tensor]:
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Unsupported data format: {type(data)}")

    if data.size(1) < sequence_length:
        raise ValueError(
            f"Data sequence length {data.size(1)} < required {sequence_length}"
        )

    data = data[:, :sequence_length]
    num_samples = data.size(0)

    def _iter():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > num_samples:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    return _iter()


def _create_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )


def _run_reference(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for _ in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = float(loss.item())

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


def _verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
) -> tuple[bool, str | None]:
    if reference.total_tokens != candidate.total_tokens:
        return (
            False,
            f"Token count mismatch: expected {reference.total_tokens}, got {candidate.total_tokens}",
        )

    ref_logits = reference.final_logits
    cand_logits = candidate.final_logits
    if isinstance(cand_logits, str):
        cand_logits = torch.load(cand_logits, weights_only=True)

    ref_logits = ref_logits.to(cand_logits.device)
    diff = (ref_logits - cand_logits).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    mean_abs = ref_logits.abs().mean().item()
    aggregate = mean_diff / mean_abs if mean_abs > 0 else mean_diff

    if aggregate > OUTPUT_VECTOR_TOLERANCE:
        return (
            False,
            (
                "Output logits mismatch: "
                f"mean_diff={mean_diff:.6f} max_diff={max_diff:.6f} "
                f"aggregate={aggregate:.6f} tol={OUTPUT_VECTOR_TOLERANCE}"
            ),
        )

    if VERIFY_LOSS:
        loss_diff = abs(reference.final_loss - candidate.final_loss)
        if loss_diff > LOSS_TOLERANCE:
            return (
                False,
                f"Loss mismatch: expected {reference.final_loss:.6f}, "
                f"got {candidate.final_loss:.6f}, tol={LOSS_TOLERANCE}",
            )

    return True, None


class Actor:
    async def evaluate(
        self,
        task_id: int,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int = 5,
        batch_size: int = 8,
        timeout: int = 600,
        sequence_length: int | None = None,
    ) -> dict:
        if not model_url or not data_url:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "missing model_url or data_url",
                "seed": seed,
            }

        code_ok, code_error = _validate_code_structure(Path(__file__).parent / "train.py")
        if not code_ok:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": code_error,
                "seed": seed,
            }

        deadline = time.monotonic() + timeout
        seed_value = abs(hash(seed)) % (2**32)
        _set_deterministic(seed_value)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = CACHE_DIR / "model"
        data_path = CACHE_DIR / "data.pt"

        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "timeout before download",
                "seed": seed,
            }

        if not model_dir.exists():
            archive_path = work_dir / "model_download"
            _download(model_url, archive_path)
            if archive_path.is_file():
                extracted = _maybe_extract(archive_path, model_dir)
                model_dir = extracted if extracted.is_dir() else model_dir

        if not data_path.exists():
            _download(data_url, data_path)

        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "timeout before eval",
                "seed": seed,
            }

        model = _get_cached_model(model_dir)
        data = _get_cached_data(data_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seq_len = sequence_length or EVAL_SEQUENCE_LENGTH
        data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
        optimizer_ref = _create_optimizer(model)

        initial_state = _CACHE.get("initial_state")
        if initial_state is None:
            initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _CACHE["initial_state"] = initial_state
        reference = _run_reference(model, data_iter_ref, optimizer_ref, steps, device)

        model.load_state_dict(initial_state)

        data_iter_miner = _create_data_iterator(data, batch_size, seq_len)
        optimizer_miner = _create_optimizer(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        miner_result = train.inner_steps(
            model=model,
            data_iterator=data_iter_miner,
            optimizer=optimizer_miner,
            num_steps=steps,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ok, error, parsed = _validate_return_type(miner_result)
        if not ok or parsed is None:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": error,
                "seed": seed,
            }

        verified, verify_error = _verify_outputs(reference, parsed)
        diagnostics = {
            "reference_loss": reference.final_loss,
            "candidate_loss": parsed.final_loss,
        }
        if reference.final_logits is not None and parsed.final_logits is not None:
            cand_logits = parsed.final_logits
            if isinstance(cand_logits, str):
                cand_logits = torch.load(cand_logits, weights_only=True)
            ref_logits = reference.final_logits.to(cand_logits.device)
            diff = (ref_logits - cand_logits).abs()
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            mean_abs = ref_logits.abs().mean().item()
            aggregate = mean_diff / mean_abs if mean_abs > 0 else mean_diff
            diagnostics.update(
                {
                    "logits_mean_diff": mean_diff,
                    "logits_max_diff": max_diff,
                    "logits_aggregate_diff": aggregate,
                }
            )
        wall_time = time.perf_counter() - start
        total_tokens = int(parsed.total_tokens)
        tps = float(total_tokens) / max(wall_time, 1e-6)

        return {
            "task_id": task_id,
            "tps": tps if verified else 0.0,
            "total_tokens": total_tokens if verified else 0,
            "wall_time_seconds": wall_time,
            "success": verified,
            "error": verify_error,
            "seed": seed,
            "diagnostics": diagnostics,
        }
"""

REQUIREMENTS_TXT = """\
torch
transformers
safetensors
accelerate
"""

DOCKERFILE = """\
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Note: HTTP server will be auto-injected by affinetes
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap affinetes env for templar TPS.")
    parser.add_argument(
        "--source-train",
        default="/home/ubuntu/templar/Chi/templar-tournament/train.py",
        help="Path to the miner train.py to package.",
    )
    parser.add_argument(
        "--output-dir",
        default="./templar-env",
        help="Directory to write the environment files.",
    )
    args = parser.parse_args()

    source_train = Path(args.source_train).resolve()
    if not source_train.exists():
        raise FileNotFoundError(f"train.py not found: {source_train}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_file(out_dir / "env.py", ENV_PY)
    write_file(out_dir / "requirements.txt", REQUIREMENTS_TXT)
    write_file(out_dir / "Dockerfile", DOCKERFILE)

    shutil.copy2(source_train, out_dir / "train.py")

    train_hash = sha256_file(out_dir / "train.py")
    commitment = {"image": "<docker-image>", "fingerprint": train_hash}

    print(f"Env written to: {out_dir}")
    print(f"train.py sha256: {train_hash}")
    print("Commitment JSON example:")
    print(json.dumps(commitment))


if __name__ == "__main__":
    main()
