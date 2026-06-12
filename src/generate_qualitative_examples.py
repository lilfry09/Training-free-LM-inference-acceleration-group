import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress_adapter import build_press, kvpress_on_gptneox
from methods import enable_gqa


DEFAULT_PROMPT = (
    "Efficient language model inference matters because long prompts require the model to"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate short qualitative examples for the report appendix.")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--model_path", type=str, default="models/pythia-70m")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--new_tokens", type=int, default=48)
    parser.add_argument("--output_json", type=str, default="outputs/qualitative_examples.json")
    parser.add_argument("--output_md", type=str, default="outputs/qualitative_examples.md")
    return parser.parse_args()


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt: str, new_tokens: int, prefill_ctx_factory=None) -> dict:
    prefill_ctx_factory = prefill_ctx_factory or (lambda _: nullcontext())
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]

    with prefill_ctx_factory(model):
        out = model(input_ids=input_ids, use_cache=True)

    past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
    generated = [int(next_token.item())]

    for step in range(1, new_tokens):
        cache_position = torch.tensor([input_ids.shape[-1] + step - 1], dtype=torch.long)
        out = model(
            input_ids=next_token.reshape(1, 1),
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
        generated.append(int(next_token.item()))

    continuation = tokenizer.decode(generated, skip_special_tokens=True)
    full_text = tokenizer.decode(torch.cat([input_ids[0], torch.tensor(generated)]), skip_special_tokens=True)
    return {
        "prompt": prompt,
        "new_tokens": new_tokens,
        "continuation": continuation,
        "full_text": full_text,
    }


def load_model(model_path: Path):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.eval()
    return model


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Qualitative Generation Examples",
        "",
        f"Prompt: `{result['prompt']}`",
        "",
    ]
    for item in result["examples"]:
        lines.extend(
            [
                f"## {item['method']}",
                "",
                "```text",
                item["continuation"].strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    model_path = (project_root / args.model_path).resolve()
    output_json = (project_root / args.output_json).resolve()
    output_md = (project_root / args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = []

    baseline = load_model(model_path)
    examples.append(
        {
            "method": "baseline",
            **greedy_generate(baseline, tokenizer, args.prompt, args.new_tokens),
        }
    )

    kv_events = []
    kvpress = load_model(model_path)
    press = build_press(method="streamingllm", compression_ratio=0.5, n_sink=4)
    examples.append(
        {
            "method": "kvpress_streamingllm_r0.5",
            **greedy_generate(
                kvpress,
                tokenizer,
                args.prompt,
                args.new_tokens,
                prefill_ctx_factory=lambda m: kvpress_on_gptneox(m, press, stats=kv_events),
            ),
            "kvpress_events": kv_events,
        }
    )

    gqa = load_model(model_path)
    enable_gqa(gqa, kv_heads=2)
    examples.append(
        {
            "method": "gqa_reduced_kv2",
            **greedy_generate(gqa, tokenizer, args.prompt, args.new_tokens),
        }
    )

    result = {
        "prompt": args.prompt,
        "new_tokens": args.new_tokens,
        "decoding": "greedy",
        "examples": examples,
    }
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, result)
    print(f"[saved] {output_json}")
    print(f"[saved] {output_md}")


if __name__ == "__main__":
    main()
