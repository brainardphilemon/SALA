"""Kaggle pipeline for SALA with Qwen2.5-7B and BLEURT-20 labels.

Official mappings are preserved:
  G3: TruthfulQA -> TriviaQA
  G5: SciQ -> NQ-Open
  G14: TruthfulQA + NQ-Open + SciQ -> TriviaQA
  G15: TruthfulQA + SciQ + TriviaQA -> NQ-Open
  G14_6SRC: TruthfulQA + NQ-Open + SciQ + PopQA + WebQuestions + HotpotQA -> TriviaQA

BLEURT-20 is the only reference-based correctness evaluator. The pipeline extracts
the final answer-token representation at every Qwen layer, matching the feature
format consumed by the original SALA implementation, trains one SALA detector per
setting, and evaluates the full target domain without target labels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_BLEURT_MODEL = "lucadiliello/BLEURT-20"
DEFAULT_BLEURT_REVISION = "e22b9eb071dcb939da9fff08b20a7e746c72a830"

SETTING_MAP = {
    "G3": {"source": "tqa", "target": "triviaqa"},
    "G5": {"source": "sciq", "target": "nq_open"},
    "G14": {"source": ["tqa", "nq_open", "sciq"], "target": "triviaqa"},
    "G15": {"source": ["tqa", "sciq", "triviaqa"], "target": "nq_open"},
    "G14_6SRC": {
        "source": ["tqa", "nq_open", "sciq", "popqa", "web_questions", "hotpotqa"],
        "target": "triviaqa",
    },
}

DATASET_SPECS = {
    "tqa": {
        "repo": "truthfulqa/truthful_qa",
        "revision": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        "config": "generation",
        "split": "validation",
    },
    "triviaqa": {
        "repo": "mandarjoshi/trivia_qa",
        "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        "config": "rc.nocontext",
        "split": "validation",
    },
    "sciq": {
        "repo": "allenai/sciq",
        "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
        "config": None,
        "split": "validation",
    },
    "nq_open": {
        "repo": "google-research-datasets/nq_open",
        "revision": "5dd9790a83002ad084ddeb7c420dc716852c6f28",
        "config": "nq_open",
        "split": "validation",
    },
    "popqa": {
        "repo": "akariasai/PopQA",
        "revision": "098765c79ea10a2cb19c828324e33281b8336ec0",
        "config": None,
        "split": "test",
    },
    "web_questions": {
        "repo": "stanfordnlp/web_questions",
        "revision": "0e473cbe21d1e91ec18da343644498be6a3f5454",
        "config": None,
        "split": "train",
    },
    "hotpotqa": {
        "repo": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "config": "distractor",
        "split": "validation",
    },
}

FINAL_COLUMNS = [
    "setting",
    "source_domain",
    "target_domain",
    "id",
    "input_text",
    "output_text",
    "ground_truth_text",
    "ground_truth_label",
    "bleurt_score",
    "raw_logit",
    "sala_probability",
    "predicted_label",
    "classifier_loss",
    "logit_adjustment_tau",
    "training_logit_adjustment",
    "source_train_prior_0",
    "source_train_prior_1",
    "split",
    "sala_threshold",
    "ground_truth_answers",
]


def default_output_dir() -> str:
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return str(kaggle / "sala_qwen25_g3_g5_bleurt")
    return "outputs/sala_qwen25_g3_g5_bleurt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument(
        "--input-artifact-dir",
        help=(
            "Read existing generation and BLEURT artifacts from this directory while writing "
            "new training results to --output-dir. Supported with --stage train."
        ),
    )
    parser.add_argument("--settings", nargs="+", choices=sorted(SETTING_MAP), default=["G3", "G5"])
    parser.add_argument("--stage", choices=["all", "generate", "bleurt", "train"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--bleurt-model", default=DEFAULT_BLEURT_MODEL)
    parser.add_argument("--bleurt-revision", default=DEFAULT_BLEURT_REVISION)
    parser.add_argument("--samples-per-domain", type=int, default=0,
                        help="Deterministic sample count for every domain; 0 uses each full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--system-prompt", default=(
        "Answer the question with only a short factual answer. Do not explain your answer."
    ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--bleurt-device", default="cuda:0")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--bleurt-batch-size", type=int, default=16)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--proj-dim", type=int, default=32)
    lodo = parser.add_mutually_exclusive_group()
    lodo.add_argument(
        "--use-lodo-dim",
        action="store_true",
        help="Select each layer's projection dimension by leave-one-source-domain-out validation",
    )
    lodo.add_argument("--no-lodo-dim", dest="use_lodo_dim", action="store_false")
    parser.set_defaults(use_lodo_dim=False)
    parser.add_argument("--proj-dim-candidates", default="8,16,32,64,128,256,512")
    parser.add_argument("--epochs-proj-selection", type=int, default=40)
    parser.add_argument("--epochs-probe-selection", type=int, default=40)
    parser.add_argument("--epochs-proj", type=int, default=40)
    parser.add_argument("--epochs-erm", type=int, default=40)
    parser.add_argument(
        "--classifier-loss",
        choices=["logit_adjusted", "bce"],
        default="logit_adjusted",
        help="Loss for the final MLP; logit_adjusted implements Menon et al. Eq. 10",
    )
    parser.add_argument(
        "--logit-adjustment-tau",
        type=float,
        default=1.0,
        help="Positive tau multiplying source-training log priors in logit-adjusted loss",
    )
    parser.add_argument("--lam-inv", type=float, default=1.0)
    parser.add_argument("--lam-sep", type=float, default=0.4)
    parser.add_argument("--lam-reg", type=float, default=0.1)
    parser.add_argument("--projection-chunk-size", type=int, default=256)
    parser.add_argument("--cache-dir")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.input_artifact_dir and args.stage != "train":
        raise ValueError("--input-artifact-dir is supported only with --stage train")
    if args.samples_per_domain < 0:
        raise ValueError("--samples-per-domain must be nonnegative")
    if args.max_new_tokens < 2 or args.max_input_tokens < 1:
        raise ValueError("Token limits are invalid")
    if args.bleurt_batch_size < 1 or args.projection_chunk_size < 1:
        raise ValueError("Batch/chunk sizes must be positive")
    if args.proj_dim < 1 or args.epochs_proj < 1 or args.epochs_erm < 1:
        raise ValueError("Projection dimension and epoch counts must be positive")
    if args.epochs_proj_selection < 1 or args.epochs_probe_selection < 1:
        raise ValueError("LODO projection/probe epoch counts must be positive")
    if args.logit_adjustment_tau <= 0:
        raise ValueError("--logit-adjustment-tau must be positive")
    candidates = [int(value.strip()) for value in args.proj_dim_candidates.split(",") if value.strip()]
    if args.use_lodo_dim and not candidates:
        raise ValueError("--use-lodo-dim requires at least one --proj-dim-candidates value")
    if any(value < 1 for value in candidates):
        raise ValueError("Projection dimension candidates must be positive")
    if not args.settings:
        raise ValueError("Select at least one SALA setting")
    for setting in args.settings:
        if args.use_lodo_dim and len(source_domains(setting)) < 2:
            raise ValueError(f"--use-lodo-dim requires multiple source domains; {setting} has one")


def json_dump_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def npz_dump_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def jsonl_dump_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def csv_dump_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=FINAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = {key: source.get(key) for key in FINAL_COLUMNS}
            row["ground_truth_answers"] = json.dumps(row["ground_truth_answers"], ensure_ascii=False)
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid/truncated JSON on {path}:{number}") from exc
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def prepare_manifest(path: Path, current: dict[str, Any]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            raise ValueError(f"Manifest mismatch at {path}; use a new --output-dir for changed settings")
    else:
        json_dump_atomic(path, current)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_texts(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _json_string_list(value: Any, domain: str, field: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{domain} field {field} is not a valid JSON list") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{domain} field {field} must be a list")
    return _unique_texts(value)


def load_domain_records(domain: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASET_SPECS[domain]
    kwargs = {
        "path": spec["repo"],
        "split": spec["split"],
        "revision": spec["revision"],
        "cache_dir": args.cache_dir,
    }
    if spec["config"] is not None:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(**kwargs)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, example in enumerate(dataset):
        if domain == "tqa":
            example_id = str(index)
            references = _unique_texts([example["best_answer"], *example["correct_answers"]])
        elif domain == "triviaqa":
            example_id = str(example["question_id"])
            if example_id in seen:
                continue
            references = _unique_texts(example["answer"]["aliases"])
        elif domain == "sciq":
            example_id = str(index)
            references = _unique_texts([example["correct_answer"]])
        elif domain == "nq_open":
            example_id = str(index)
            references = _unique_texts(example["answer"])
        elif domain == "popqa":
            example_id = str(example["id"])
            references = _json_string_list(example["possible_answers"], domain, "possible_answers")
        elif domain == "web_questions":
            example_id = str(index)
            references = _unique_texts(example["answers"])
        elif domain == "hotpotqa":
            example_id = str(example["id"])
            references = _unique_texts([example["answer"]])
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        seen.add(example_id)
        if not references:
            raise ValueError(f"{domain}:{example_id} has no reference answers")
        records.append({
            "id": example_id,
            "input_text": str(example["question"]).strip(),
            "ground_truth_answers": references,
        })
    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    if args.samples_per_domain:
        order = order[: min(args.samples_per_domain, len(order))]
    return [records[index] for index in order]


def required_domains(settings: Sequence[str]) -> list[str]:
    domains: list[str] = []
    for setting in settings:
        mapping = SETTING_MAP[setting]
        setting_domains = [*source_domains(setting), str(mapping["target"])]
        for domain in setting_domains:
            if domain not in domains:
                domains.append(domain)
    return domains


def source_domains(setting: str) -> list[str]:
    """Return the ordered source environments for a SALA setting."""
    source = SETTING_MAP[setting]["source"]
    if isinstance(source, str):
        return [source]
    return [str(domain) for domain in source]


def chat_prompt(tokenizer, question: str, system_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def compute_dtype(torch, device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def release_gpu(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generation_manifest(
    domain: str,
    records: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    num_layers: int,
    hidden_size: int,
) -> dict[str, Any]:
    spec = DATASET_SPECS[domain]
    return {
        "schema_version": 1,
        "domain": domain,
        "dataset": spec,
        "selected_ids_sha256": sha256_json([row["id"] for row in records]),
        "selected_count": len(records),
        "model": args.model,
        "model_revision": args.model_revision,
        "four_bit": not args.no_4bit,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "system_prompt": args.system_prompt,
        "feature_definition": "last non-EOS answer token hidden state at embedding output plus every transformer layer",
        "feature_shape": [len(records), num_layers + 1, hidden_size],
        "feature_dtype": "float16",
    }


def trim_terminal_tokens(generated_ids, tokenizer):
    terminal = {value for value in [tokenizer.eos_token_id, tokenizer.pad_token_id] if value is not None}
    end = len(generated_ids)
    while end and int(generated_ids[end - 1]) in terminal:
        end -= 1
    return generated_ids[:end]


def stack_last_token_features(hidden_states, torch):
    """Stack embedding/output-layer states at the final replayed answer token."""
    return torch.stack([hidden[0, -1].float().cpu() for hidden in hidden_states], dim=0)


def ensure_generation(
    records_by_domain: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Kaggle GPU accelerator")
    if not args.no_4bit and not str(args.device).startswith("cuda"):
        raise ValueError("4-bit bitsandbytes loading requires CUDA; use --no-4bit for CPU")
    config = AutoConfig.from_pretrained(
        args.model, revision=args.model_revision, cache_dir=args.cache_dir
    )
    incomplete: list[str] = []
    for domain, records in records_by_domain.items():
        manifest = generation_manifest(
            domain, records, args, config.num_hidden_layers, config.hidden_size
        )
        manifest_path = output_dir / "generation" / domain / "manifest.json"
        prepare_manifest(manifest_path, manifest)
        rows = read_jsonl(output_dir / "generation" / domain / "rows.jsonl")
        expected_ids = {str(row["id"]) for row in records}
        stored_ids = {str(row["id"]) for row in rows}
        if not stored_ids.issubset(expected_ids):
            raise ValueError(f"Stored rows do not belong to the selected {domain} cohort")
        feature_path = output_dir / "generation" / domain / "features.npy"
        if len(rows) < len(records):
            incomplete.append(domain)
        elif len(rows) > len(records):
            raise ValueError(f"Too many generated rows for {domain}")
        elif not feature_path.is_file():
            raise FileNotFoundError(f"Completed row log exists but features are missing: {feature_path}")
    if not incomplete:
        print("Generation/features already complete", flush=True)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, cache_dir=args.cache_dir, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = compute_dtype(torch, args.device)
    load_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "cache_dir": args.cache_dir,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if not args.no_4bit:
        load_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": args.device},
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.no_4bit:
        model = model.to(args.device)
    model.eval()
    try:
        for domain in incomplete:
            records = records_by_domain[domain]
            domain_dir = output_dir / "generation" / domain
            rows_path = domain_dir / "rows.jsonl"
            feature_path = domain_dir / "features.npy"
            saved_rows = read_jsonl(rows_path)
            completed = {str(row["id"]) for row in saved_rows}
            expected_ids = {str(row["id"]) for row in records}
            if not completed.issubset(expected_ids):
                raise ValueError(f"Stored rows do not belong to the selected {domain} cohort")
            shape = (len(records), model.config.num_hidden_layers + 1, model.config.hidden_size)
            if feature_path.exists():
                features = np.lib.format.open_memmap(feature_path, mode="r+")
                if features.shape != shape or features.dtype != np.float16:
                    raise ValueError(f"Feature array mismatch for {domain}: {features.shape}/{features.dtype}")
            else:
                if saved_rows:
                    raise FileNotFoundError(f"{feature_path} is missing but resumable rows exist")
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=shape)
            with rows_path.open("a", encoding="utf-8") as stream, torch.inference_mode():
                for position, record in enumerate(records):
                    example_id = str(record["id"])
                    if example_id in completed:
                        continue
                    prompt = chat_prompt(tokenizer, record["input_text"], args.system_prompt)
                    encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.max_input_tokens,
                    ).to(args.device)
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )[0, encoded["input_ids"].shape[1]:]
                    answer_ids = trim_terminal_tokens(generated, tokenizer)
                    replay_ids = torch.cat([encoded["input_ids"][0], answer_ids]).unsqueeze(0)
                    replay = model(
                        replay_ids,
                        attention_mask=torch.ones_like(replay_ids),
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    layerwise = stack_last_token_features(replay.hidden_states, torch).numpy().astype(np.float16)
                    features[position] = layerwise
                    features.flush()
                    row = {
                        "schema_version": 1,
                        "id": example_id,
                        "position": position,
                        "domain": domain,
                        "input_text": record["input_text"],
                        "model_input_text": prompt,
                        "output_text": tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
                        "ground_truth_answers": record["ground_truth_answers"],
                        "input_length": int(encoded["input_ids"].shape[1]),
                        "output_length": int(len(answer_ids)),
                        "feature_row": position,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed.add(example_id)
                    del generated, answer_ids, replay_ids, replay, layerwise, encoded
                    print(f"generate {domain} {position + 1}/{len(records)} id={example_id}", flush=True)
            del features
    finally:
        del model, tokenizer
        release_gpu(torch)


class Bleurt20Scorer:
    def __init__(self, args: argparse.Namespace):
        import torch
        from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
        from huggingface_hub import hf_hub_download

        if str(args.bleurt_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("BLEURT CUDA device requested but unavailable")
        self.torch = torch
        self.device = args.bleurt_device
        self.batch_size = args.bleurt_batch_size
        self.model_name = args.bleurt_model
        self.revision = args.bleurt_revision
        self.tokenizer = BleurtTokenizer.from_pretrained(
            args.bleurt_model, revision=args.bleurt_revision, cache_dir=args.cache_dir
        )
        config = BleurtConfig.from_pretrained(
            args.bleurt_model, revision=args.bleurt_revision, cache_dir=args.cache_dir
        )
        local_model = Path(args.bleurt_model).expanduser()
        if local_model.is_dir():
            weights_path = local_model / "pytorch_model.bin"
        else:
            weights_path = Path(hf_hub_download(
                repo_id=args.bleurt_model,
                filename="pytorch_model.bin",
                revision=args.bleurt_revision,
                cache_dir=args.cache_dir,
            ))
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("BLEURT-20 requires a PyTorch version with weights_only support") from exc
        self.model = BleurtForSequenceClassification(config)
        self.model.load_state_dict(state_dict, strict=True)
        del state_dict
        self.model.eval()
        if str(self.device).startswith("cuda"):
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

    def close(self) -> None:
        del self.model, self.tokenizer
        release_gpu(self.torch)

    def score_rows(self, rows: Sequence[dict[str, Any]], cache_path: Path) -> list[dict[str, Any]]:
        cached_rows = read_jsonl(cache_path)
        cached = {str(row["id"]): row for row in cached_rows}
        expected = {str(row["id"]) for row in rows}
        if not set(cached).issubset(expected):
            raise ValueError(f"Cached BLEURT rows do not belong to {cache_path}")
        if len(cached) == len(rows):
            return [cached[str(row["id"])] for row in rows]
        references: list[str] = []
        candidates: list[str] = []
        owners: list[tuple[int, int]] = []
        for row_index, row in enumerate(rows):
            if str(row["id"]) in cached:
                continue
            aliases = row.get("ground_truth_answers") or []
            if not aliases:
                raise ValueError(f"Row {row.get('id')} has no references")
            for reference_index, reference in enumerate(aliases):
                references.append(str(reference))
                candidates.append(str(row["output_text"]))
                owners.append((row_index, reference_index))
        last_pair: dict[int, int] = {}
        for pair_index, (row_index, _) in enumerate(owners):
            last_pair[row_index] = pair_index
        best: dict[int, tuple[float, str, int]] = {}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as stream, self.torch.inference_mode():
            for start in range(0, len(references), self.batch_size):
                stop = min(start + self.batch_size, len(references))
                encoded = self.tokenizer(
                    references[start:stop],
                    candidates[start:stop],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                scores = self.model(**encoded).logits.reshape(-1).float().cpu().tolist()
                for pair_index, score_value in enumerate(scores, start):
                    score = float(score_value)
                    row_index, reference_index = owners[pair_index]
                    if not math.isfinite(score):
                        raise ValueError(f"Non-finite BLEURT score for {rows[row_index]['id']}")
                    reference = str(rows[row_index]["ground_truth_answers"][reference_index])
                    current = best.get(row_index)
                    if current is None or score > current[0]:
                        best[row_index] = (score, reference, reference_index)
                    if pair_index == last_pair[row_index]:
                        selected = best[row_index]
                        saved = {
                            "id": str(rows[row_index]["id"]),
                            "bleurt_score": selected[0],
                            "ground_truth_text": selected[1],
                            "ground_truth_reference_index": selected[2],
                            "bleurt_model": self.model_name,
                            "bleurt_revision": self.revision,
                        }
                        stream.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        cached[saved["id"]] = saved
                print(f"BLEURT {cache_path.parent.name} pairs {stop}/{len(references)}", flush=True)
        if len(cached) != len(rows):
            raise RuntimeError(f"BLEURT did not score every row for {cache_path}")
        return [cached[str(row["id"])] for row in rows]


def ensure_bleurt(domains: Sequence[str], args: argparse.Namespace, output_dir: Path) -> None:
    pending: list[tuple[str, list[dict[str, Any]]]] = []
    for domain in domains:
        rows = read_jsonl(output_dir / "generation" / domain / "rows.jsonl")
        manifest = {
            "schema_version": 1,
            "domain": domain,
            "cohort_sha256": sha256_json([row["id"] for row in rows]),
            "bleurt_model": args.bleurt_model,
            "bleurt_revision": args.bleurt_revision,
            "precision": "float16" if str(args.bleurt_device).startswith("cuda") else "float32",
            "reference_policy": "maximum score over all aliases",
        }
        prepare_manifest(output_dir / "bleurt" / domain / "manifest.json", manifest)
        scored = read_jsonl(output_dir / "bleurt" / domain / "scores.jsonl")
        expected_ids = {str(row["id"]) for row in rows}
        scored_ids = {str(row["id"]) for row in scored}
        if not scored_ids.issubset(expected_ids):
            raise ValueError(f"BLEURT rows do not belong to the generated {domain} cohort")
        if len(scored) < len(rows):
            pending.append((domain, rows))
        elif len(scored) > len(rows):
            raise ValueError(f"Too many BLEURT rows for {domain}")
    if not pending:
        print("BLEURT scoring already complete", flush=True)
        return
    scorer = Bleurt20Scorer(args)
    try:
        for domain, rows in pending:
            scorer.score_rows(rows, output_dir / "bleurt" / domain / "scores.jsonl")
    finally:
        scorer.close()


def calibrate_threshold(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, dict[str, Any]]:
    if not scores or len(scores) != len(labels):
        raise ValueError("Threshold calibration requires equal nonempty scores and labels")
    unique = sorted(set(float(score) for score in scores))
    candidates = [math.nextafter(unique[0], -math.inf)]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.append(math.nextafter(unique[-1], math.inf))
    positives = sum(labels)
    negatives = len(labels) - positives
    choices = []
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
        tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
        balanced = 0.5 * (tp / positives + tn / negatives) if positives and negatives else (tp + tn) / len(labels)
        accuracy = sum(y == p for y, p in zip(labels, predictions)) / len(labels)
        gap = abs(sum(predictions) / len(predictions) - positives / len(labels))
        choices.append((balanced, accuracy, -gap, -threshold, threshold))
    selected = max(choices)
    return float(selected[-1]), {
        "objective": "maximum source-validation balanced accuracy",
        "balanced_accuracy": selected[0],
        "accuracy": selected[1],
        "n": len(labels),
        "n_correct": int(sum(labels)),
        "has_both_classes": len(set(labels)) == 2,
    }


def binary_metrics(labels: Sequence[int], predictions: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    if not labels:
        return {"n": 0}
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    balanced = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else None
    return {
        "n": len(labels),
        "n_correct": int(sum(labels)),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auc,
        "confusion_matrix": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
    }


def predict_scores(model, features: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    logits = predict_logits(model, features, device, batch_size=batch_size)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def predict_logits(model, features: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    import torch

    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).float().to(device)
            output.append(model(batch).float().cpu().numpy())
    return np.concatenate(output).reshape(-1)


def project_target_layer(
    features,
    layer: int,
    mean: np.ndarray,
    std: np.ndarray,
    projection: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    result = np.empty((features.shape[0], projection.shape[1]), dtype=np.float32)
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        block = np.asarray(features[start:stop, layer, :], dtype=np.float32)
        result[start:stop] = (((block - mean) / std) @ projection - z_mean) / z_std
    return result


def training_manifest(setting: str, args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    mapping = SETTING_MAP[setting]
    manifests = {}
    domains = [*source_domains(setting), str(mapping["target"])]
    for domain in domains:
        score_path = artifact_dir / "bleurt" / domain / "scores.jsonl"
        manifests[domain] = {
            "generation": json.loads((artifact_dir / "generation" / domain / "manifest.json").read_text()),
            "bleurt": json.loads((artifact_dir / "bleurt" / domain / "manifest.json").read_text()),
            "bleurt_scores_sha256": file_sha256(score_path),
        }
    return {
        "schema_version": 2,
        "setting": setting,
        "mapping": mapping,
        "inputs_sha256": sha256_json(manifests),
        "bleurt_threshold": args.bleurt_threshold,
        "proj_dim": args.proj_dim,
        "use_lodo_dim": args.use_lodo_dim,
        "proj_dim_candidates": args.proj_dim_candidates,
        "epochs_proj_selection": args.epochs_proj_selection,
        "epochs_probe_selection": args.epochs_probe_selection,
        "epochs_proj": args.epochs_proj,
        "epochs_erm": args.epochs_erm,
        "classifier_loss": args.classifier_loss,
        "logit_adjustment_tau": args.logit_adjustment_tau,
        "lam_inv": args.lam_inv,
        "lam_sep": args.lam_sep,
        "lam_reg": args.lam_reg,
        "seed": args.seed,
        "source_split": "per-domain 67.5% train, 7.5% validation, 25% source test; stratified",
        "target_policy": "full target domain; no target labels used for fitting or model selection",
    }


def load_labels(domain: str, args: argparse.Namespace, artifact_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = read_jsonl(artifact_dir / "generation" / domain / "rows.jsonl")
    scored = read_jsonl(artifact_dir / "bleurt" / domain / "scores.jsonl")
    by_id = {str(row["id"]): row for row in scored}
    if len(by_id) != len(rows):
        raise ValueError(f"Incomplete BLEURT scores for {domain}")
    labels = np.asarray(
        [float(by_id[str(row["id"])]["bleurt_score"]) >= args.bleurt_threshold for row in rows],
        dtype=np.int64,
    )
    return rows, labels


def source_split_indices(labels: np.ndarray, setting: str, domain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match SALA's independent stratified train/validation/test split per source domain."""
    from sklearn.model_selection import StratifiedShuffleSplit

    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"{setting} source {domain} contains only one BLEURT class; "
            "increase --samples-per-domain or use the full dataset"
        )
    indices = np.arange(len(labels))
    try:
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        trainval_idx, test_idx = next(outer.split(indices, labels))
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=43)
        train_pos, val_pos = next(inner.split(trainval_idx, labels[trainval_idx]))
    except ValueError as exc:
        raise ValueError(
            f"{setting} source {domain} is too small for the stratified SALA split; "
            "increase --samples-per-domain"
        ) from exc
    return trainval_idx[train_pos], trainval_idx[val_pos], test_idx


def train_setting(
    setting: str,
    args: argparse.Namespace,
    output_dir: Path,
    artifact_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from models.projections import compute_isr_init_W, train_layerwise_invariant_projection
    from utils.helpers import (
        binary_logit_adjustment,
        seed_everything,
        select_best_proj_dim_lodo,
        train_erm,
    )

    artifact_dir = output_dir if artifact_dir is None else artifact_dir
    setting_dir = output_dir / "results" / setting
    current_manifest = training_manifest(setting, args, artifact_dir)
    prepare_manifest(setting_dir / "manifest.json", current_manifest)
    result_path = setting_dir / "results.jsonl"
    summary_path = setting_dir / "summary.json"
    if result_path.exists() and summary_path.exists():
        return read_jsonl(result_path), json.loads(summary_path.read_text(encoding="utf-8"))

    mapping = SETTING_MAP[setting]
    sources = source_domains(setting)
    target_domain = str(mapping["target"])
    target_rows, target_labels = load_labels(target_domain, args, artifact_dir)
    target_features = np.load(
        artifact_dir / "generation" / target_domain / "features.npy", mmap_mode="r"
    )

    source_bundles: list[dict[str, Any]] = []
    for domain_id, domain in enumerate(sources):
        rows, labels = load_labels(domain, args, artifact_dir)
        features = np.load(artifact_dir / "generation" / domain / "features.npy", mmap_mode="r")
        if features.shape[1:] != target_features.shape[1:]:
            raise ValueError(f"Feature shapes differ between {domain} and {target_domain}")
        train_idx, val_idx, test_idx = source_split_indices(labels, setting, domain)
        source_bundles.append({
            "domain": domain,
            "domain_id": domain_id,
            "rows": rows,
            "labels": labels,
            "features": features,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        })

    num_layers, hidden_size = target_features.shape[1:]
    if args.proj_dim > hidden_size:
        raise ValueError(f"--proj-dim {args.proj_dim} exceeds hidden size {hidden_size}")

    y_train = np.concatenate([bundle["labels"][bundle["train_idx"]] for bundle in source_bundles])
    y_val = np.concatenate([bundle["labels"][bundle["val_idx"]] for bundle in source_bundles])
    y_source_test = np.concatenate([bundle["labels"][bundle["test_idx"]] for bundle in source_bundles])
    domain_ids = np.concatenate([
        np.full(len(bundle["train_idx"]), bundle["domain_id"], dtype=np.int64)
        for bundle in source_bundles
    ])
    val_domain_ids = np.concatenate([
        np.full(len(bundle["val_idx"]), bundle["domain_id"], dtype=np.int64)
        for bundle in source_bundles
    ])
    for bundle in source_bundles:
        train_labels = bundle["labels"][bundle["train_idx"]]
        if min(np.bincount(train_labels, minlength=2)) < 2:
            raise ValueError(
                f"{setting} source {bundle['domain']} needs at least two training examples per BLEURT class"
            )

    device = torch.device(args.train_device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    z_train_layers: list[np.ndarray] = []
    z_val_layers: list[np.ndarray] = []
    z_source_test_layers: list[np.ndarray] = []
    z_target_layers: list[np.ndarray] = []
    selected_proj_dims: list[int] = []
    lodo_selection: list[dict[str, Any]] = []
    candidate_dims = [
        int(value.strip())
        for value in args.proj_dim_candidates.split(",")
        if value.strip() and int(value.strip()) <= hidden_size
    ]
    if args.use_lodo_dim and not candidate_dims:
        raise ValueError("No --proj-dim-candidates fit the model hidden size")

    checkpoint_dir = setting_dir / "projection_checkpoints"
    prepare_manifest(checkpoint_dir / "manifest.json", current_manifest)

    for layer in range(num_layers):
        checkpoint_path = checkpoint_dir / f"layer_{layer:02d}.npz"
        if checkpoint_path.exists():
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                saved_layer = int(np.asarray(checkpoint["layer"]).item())
                if saved_layer != layer:
                    raise ValueError(
                        f"Projection checkpoint layer mismatch at {checkpoint_path}: {saved_layer}"
                    )
                projection_dim = int(np.asarray(checkpoint["projection_dim"]).item())
                selection_score = float(np.asarray(checkpoint["selection_score"]).item())
                z_train_layers.append(np.asarray(checkpoint["z_train"], dtype=np.float32))
                z_val_layers.append(np.asarray(checkpoint["z_val"], dtype=np.float32))
                z_source_test_layers.append(
                    np.asarray(checkpoint["z_source_test"], dtype=np.float32)
                )
                z_target_layers.append(np.asarray(checkpoint["z_target"], dtype=np.float32))
            selected_proj_dims.append(projection_dim)
            if args.use_lodo_dim:
                lodo_selection.append({
                    "layer": layer,
                    "projection_dim": projection_dim,
                    "selection_score": selection_score if math.isfinite(selection_score) else None,
                })
            print(
                f"{setting} resumed projected layer {layer + 1}/{num_layers} dim={projection_dim}",
                flush=True,
            )
            continue

        x_train = np.concatenate([
            np.asarray(bundle["features"][bundle["train_idx"], layer, :], dtype=np.float32)
            for bundle in source_bundles
        ])
        x_val = np.concatenate([
            np.asarray(bundle["features"][bundle["val_idx"], layer, :], dtype=np.float32)
            for bundle in source_bundles
        ])
        x_source_test = np.concatenate([
            np.asarray(bundle["features"][bundle["test_idx"], layer, :], dtype=np.float32)
            for bundle in source_bundles
        ])
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-8
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_source_test = (x_source_test - mean) / std

        projection_dim = args.proj_dim
        if args.use_lodo_dim:
            projection_dim, lodo_score, _ = select_best_proj_dim_lodo(
                x_train,
                y_train,
                domain_ids,
                x_val,
                y_val,
                val_domain_ids,
                proj_dim_candidates=candidate_dims,
                layer=layer,
                args=args,
                device=device,
                lam_inv=args.lam_inv,
                lam_sep=args.lam_sep,
                lam_reg=args.lam_reg,
                epochs_proj=args.epochs_proj_selection,
                epochs_probe=args.epochs_probe_selection,
            )
            if projection_dim is None:
                raise RuntimeError(f"LODO projection selection failed for {setting} layer {layer}")
            lodo_selection.append({
                "layer": layer,
                "projection_dim": int(projection_dim),
                "selection_score": float(lodo_score) if math.isfinite(float(lodo_score)) else None,
            })
        selected_proj_dims.append(int(projection_dim))

        seed_everything(args.seed + 1000 + (layer + 1) * 10000)
        init_w = compute_isr_init_W(
            x_train,
            y_train,
            domain_ids,
            proj_dim=projection_dim,
            device=device,
        )
        projection = train_layerwise_invariant_projection(
            x_train,
            y_train,
            domain_ids,
            init_W=init_w,
            proj_dim=projection_dim,
            n_epochs=args.epochs_proj,
            lr=1e-3,
            lam_inv=args.lam_inv,
            lam_sep=args.lam_sep,
            lam_reg=args.lam_reg,
            device=device,
            verbose=False,
        )
        z_train = x_train @ projection
        z_mean = z_train.mean(axis=0, keepdims=True)
        z_std = z_train.std(axis=0, keepdims=True) + 1e-6
        z_train_layer = (z_train - z_mean) / z_std
        z_val_layer = (x_val @ projection - z_mean) / z_std
        z_source_test_layer = (x_source_test @ projection - z_mean) / z_std
        z_target_layer = project_target_layer(
            target_features,
            layer,
            mean,
            std,
            projection,
            z_mean,
            z_std,
            args.projection_chunk_size,
        )
        selection_score = (
            float(lodo_selection[-1]["selection_score"])
            if args.use_lodo_dim and lodo_selection[-1]["selection_score"] is not None
            else float("nan")
        )
        npz_dump_atomic(
            checkpoint_path,
            layer=np.asarray(layer, dtype=np.int64),
            projection_dim=np.asarray(projection_dim, dtype=np.int64),
            selection_score=np.asarray(selection_score, dtype=np.float64),
            z_train=np.asarray(z_train_layer, dtype=np.float32),
            z_val=np.asarray(z_val_layer, dtype=np.float32),
            z_source_test=np.asarray(z_source_test_layer, dtype=np.float32),
            z_target=np.asarray(z_target_layer, dtype=np.float32),
        )
        z_train_layers.append(z_train_layer)
        z_val_layers.append(z_val_layer)
        z_source_test_layers.append(z_source_test_layer)
        z_target_layers.append(z_target_layer)
        print(
            f"{setting} projected layer {layer + 1}/{num_layers} dim={projection_dim}",
            flush=True,
        )

    z_train_all = np.concatenate(z_train_layers, axis=1)
    z_val_all = np.concatenate(z_val_layers, axis=1)
    z_source_test_all = np.concatenate(z_source_test_layers, axis=1)
    z_target_all = np.concatenate(z_target_layers, axis=1)
    concat_dim = int(z_train_all.shape[1])

    seed_everything(args.seed)
    model, best_source_val_auc = train_erm(
        z_train_all,
        y_train,
        z_val_all,
        y_val,
        device=device,
        epochs=args.epochs_erm,
        batch_size=128,
        lr=1e-3,
        wd=1e-4,
        verbose=False,
        loss_name=args.classifier_loss,
        logit_adjustment_tau=args.logit_adjustment_tau,
    )
    if args.classifier_loss == "logit_adjusted":
        training_logit_adjustment, loss_details = binary_logit_adjustment(
            y_train, tau=args.logit_adjustment_tau
        )
    else:
        training_logit_adjustment = 0.0
        counts = np.bincount(y_train, minlength=2).astype(np.int64)
        priors = counts.astype(np.float64) / counts.sum()
        loss_details = {
            "tau": None,
            "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
            "class_priors": {"0": float(priors[0]), "1": float(priors[1])},
            "positive_logit_adjustment": 0.0,
            "training_formula": "BCEWithLogits(raw_logit, label)",
            "inference_formula": "sala_probability = sigmoid(raw_logit)",
        }
    val_scores = predict_scores(model, z_val_all, device)
    source_test_scores = predict_scores(model, z_source_test_all, device)
    target_logits = predict_logits(model, z_target_all, device)
    target_scores = 1.0 / (1.0 + np.exp(-np.clip(target_logits, -80.0, 80.0)))
    if args.classifier_loss == "logit_adjusted":
        threshold = 0.5
        threshold_summary = {
            "objective": "Menon et al. Eq. 10 inference: argmax of unadjusted logits",
            "probability_threshold": threshold,
            "source_validation_used_for_threshold": False,
        }
    else:
        threshold, threshold_summary = calibrate_threshold(val_scores.tolist(), y_val.tolist())
    source_test_predictions = (source_test_scores >= threshold).astype(np.int64)
    target_predictions = (target_scores >= threshold).astype(np.int64)

    target_bleurt = {
        str(row["id"]): row
        for row in read_jsonl(artifact_dir / "bleurt" / target_domain / "scores.jsonl")
    }
    results = []
    priors = loss_details["class_priors"]
    for row, label, raw_logit, score, prediction in zip(
        target_rows, target_labels, target_logits, target_scores, target_predictions
    ):
        judged = target_bleurt[str(row["id"])]
        results.append({
            **row,
            "setting": setting,
            "source_domain": "+".join(sources),
            "source_domains": sources,
            "target_domain": target_domain,
            "ground_truth_text": judged["ground_truth_text"],
            "ground_truth_label": int(label),
            "bleurt_score": float(judged["bleurt_score"]),
            "raw_logit": float(raw_logit),
            "sala_probability": float(score),
            "predicted_label": int(prediction),
            "classifier_loss": args.classifier_loss,
            "logit_adjustment_tau": (
                args.logit_adjustment_tau if args.classifier_loss == "logit_adjusted" else None
            ),
            "training_logit_adjustment": training_logit_adjustment,
            "source_train_prior_0": priors["0"],
            "source_train_prior_1": priors["1"],
            "split": "target_evaluation",
            "sala_threshold": threshold,
        })
    summary = {
        "schema_version": 2,
        "setting": setting,
        "source_domain": "+".join(sources),
        "source_domains": sources,
        "target_domain": target_domain,
        "model": args.model,
        "bleurt_model": args.bleurt_model,
        "bleurt_threshold": args.bleurt_threshold,
        "label_definition": "1 = correct/non-hallucinated when max BLEURT-20 over aliases >= threshold; 0 otherwise",
        "label_warning": "BLEURT labels are automatic pseudo-labels, not human correctness annotations.",
        "sala_probability_definition": "sigmoid of the final MLP raw, unadjusted inference logit for BLEURT-correct/non-hallucinated",
        "classifier_loss": args.classifier_loss,
        "logit_adjustment": loss_details,
        "logit_adjustment_reference": "Menon et al., Long-Tail Learning via Logit Adjustment, arXiv:2007.07314, Eq. 10",
        "feature_extraction_changed": False,
        "sala_threshold": threshold,
        "threshold_calibration": threshold_summary,
        "best_source_validation_auroc_during_training": best_source_val_auc,
        "source_test_metrics": binary_metrics(
            y_source_test.tolist(), source_test_predictions.tolist(), source_test_scores.tolist()
        ),
        "target_metrics": binary_metrics(
            target_labels.tolist(), target_predictions.tolist(), target_scores.tolist()
        ),
        "target_supervision_used": False,
        "source_counts": {
            bundle["domain"]: {
                "total": len(bundle["labels"]),
                "train": len(bundle["train_idx"]),
                "validation": len(bundle["val_idx"]),
                "test": len(bundle["test_idx"]),
            }
            for bundle in source_bundles
        },
        "projection_dimension_policy": "lodo" if args.use_lodo_dim else "fixed",
        "selected_projection_dimensions": selected_proj_dims,
        "lodo_selection": lodo_selection,
        "num_layers_including_embedding_output": num_layers,
        "hidden_size": hidden_size,
        "projected_concat_dimension": concat_dim,
    }
    jsonl_dump_atomic(result_path, results)
    csv_dump_atomic(setting_dir / "results.csv", results)
    json_dump_atomic(summary_path, summary)
    del model, target_features
    for bundle in source_bundles:
        del bundle["features"]
    release_gpu(torch)
    return results, summary


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = (
        Path(args.input_artifact_dir).expanduser().resolve()
        if args.input_artifact_dir
        else output_dir
    )
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Input artifact directory does not exist: {artifact_dir}")
    domains = required_domains(args.settings)
    if args.stage in {"all", "generate"}:
        records_by_domain = {domain: load_domain_records(domain, args) for domain in domains}
        ensure_generation(records_by_domain, args, output_dir)
    if args.stage == "generate":
        return
    if args.stage in {"all", "bleurt"}:
        ensure_bleurt(domains, args, output_dir)
    if args.stage == "bleurt":
        return

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for setting in args.settings:
        rows, summary = train_setting(setting, args, output_dir, artifact_dir=artifact_dir)
        all_rows.extend(rows)
        summaries[setting] = summary
    jsonl_dump_atomic(output_dir / "results.jsonl", all_rows)
    csv_dump_atomic(output_dir / "results.csv", all_rows)
    json_dump_atomic(output_dir / "summary.json", summaries)
    print(json.dumps(summaries, indent=2, allow_nan=False), flush=True)
    print(f"Combined results: {output_dir / 'results.csv'}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    from contextlib import contextmanager
    import fcntl

    @contextmanager
    def run_lock(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another SALA pipeline is writing {args.output_dir}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    with run_lock(Path(args.output_dir).expanduser().resolve() / "pipeline.lock"):
        run(args)


if __name__ == "__main__":
    main()
