import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local model and dataset assets for the experiments.")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--model_dir", type=str, default="models/pythia-70m")
    parser.add_argument("--wikitext_dir", type=str, default="datasets/wikitext-103-raw-v1")
    parser.add_argument("--pg19_dir", type=str, default="datasets/pg19_samples")
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_wikitext", action="store_true")
    parser.add_argument("--skip_pg19", action="store_true")
    return parser.parse_args()


def prepare_model(project_root: Path, model_name: str, model_dir: str) -> None:
    output_dir = project_root / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"[saved] {output_dir}")


def prepare_wikitext(project_root: Path, wikitext_dir: str) -> None:
    output_dir = project_root / wikitext_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    dataset.save_to_disk(output_dir)
    print(f"[saved] {output_dir}")


def _read_pg19_text(example: dict) -> str:
    if "text" in example:
        return example["text"]
    if "book_text" in example:
        return example["book_text"]
    raise KeyError(f"Could not find text field in PG-19 example keys: {sorted(example.keys())}")


def prepare_pg19(project_root: Path, pg19_dir: str) -> None:
    output_dir = project_root / pg19_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        stream = load_dataset("pg19", split=split, streaming=True, trust_remote_code=True)
        example = next(iter(stream))
        text = _read_pg19_text(example)
        text_path = output_dir / f"{split}_1.txt"
        meta_path = output_dir / f"{split}_1.meta.json"
        text_path.write_text(text, encoding="utf-8")
        meta = {key: value for key, value in example.items() if key not in {"text", "book_text"}}
        meta["chars"] = len(text)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {text_path}")


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if not args.skip_model:
        prepare_model(project_root, args.model_name, args.model_dir)
    if not args.skip_wikitext:
        prepare_wikitext(project_root, args.wikitext_dir)
    if not args.skip_pg19:
        prepare_pg19(project_root, args.pg19_dir)


if __name__ == "__main__":
    main()
