#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate template prefixes and compute log-odds for "he" vs "she" next-token probabilities.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

def past_tense(verb):
    if verb.endswith('e'):
        return verb[:-1] + 'ed'
    else:
        return verb + 'ed'


def present_participle(verb):
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith(("ee", "ye")):
        return verb[:-1] + "ing"
    return verb + "ing"


def format_template(template, profession, verb):
    return template.format(
        det1=get_det(profession, capitalise=True),
        person_1=profession,
        verb=verb,
        verb_ed=past_tense(verb),
        verb_ing=present_participle(verb),
    )


def verb_forms_for_template(template, verb):
    forms = set()
    if "{verb}" in template:
        forms.add(verb)
    if "{verb_ed}" in template:
        forms.add(past_tense(verb))
    if "{verb_ing}" in template:
        forms.add(present_participle(verb))
    return forms


def generate_verb_forms() -> list[str]:
    forms = set()
    for template, _, _, _ in templates:
        for verb, _ in verbs_ratings:
            forms.update(verb_forms_for_template(template, verb))
    return sorted(forms)


def write_verb_forms(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(generate_verb_forms()) + "\n", encoding="utf-8")


def get_det(noun, capitalise=False):
    if noun[0].lower() in ('a', 'e', 'i', 'o', 'u'):
        return 'An' if capitalise else 'an'
    else:
        return 'A' if capitalise else 'a'

templates = [
    ("{det1} {person_1} {verb_ed} a parent. As for the {person_1},", "simple past", "subject", "agent"),
    ("{det1} {person_1} was {verb_ed} by a parent. As for the {person_1},", "simple past", "subject", "patient"),
    ("A parent {verb_ed} {det1} {person_1}. As for the {person_1},", "simple past", "non-subject", "patient"),
    ("A parent was {verb_ed} by {det1} {person_1}. As for the {person_1},", "simple past", "non-subject", "agent"),

    ("{det1} {person_1} will {verb} a parent. As for the {person_1},", "simple future", "subject", "agent"),
    ("{det1} {person_1} will be {verb_ed} by a parent. As for the {person_1},", "simple future", "subject", "patient"),
    ("A parent will {verb} {det1} {person_1}. As for the {person_1},", "simple future", "non-subject", "patient"),
    ("A parent will be {verb_ed} by {det1} {person_1}. As for the {person_1},", "simple future", "non-subject", "agent"),

    ("{det1} {person_1} is {verb_ing} a parent. As for the {person_1},", "present continuous", "subject", "agent"),
    ("{det1} {person_1} is being {verb_ed} by a parent. As for the {person_1},", "present continuous", "subject", "patient"),
    ("A parent is {verb_ing} {det1} {person_1}. As for the {person_1},", "present continuous", "non-subject", "patient"),
    ("A parent is being {verb_ed} by {det1} {person_1}. As for the {person_1},", "present continuous", "non-subject", "agent"),

    ("{det1} {person_1} was {verb_ing} a parent. As for the {person_1},", "past continuous", "subject", "agent"),
    ("{det1} {person_1} was being {verb_ed} by a parent. As for the {person_1},", "past continuous", "subject", "patient"),
    ("A parent was {verb_ing} {det1} {person_1}. As for the {person_1},", "past continuous", "non-subject", "patient"),
    ("A parent was being {verb_ed} by {det1} {person_1}. As for the {person_1},", "past continuous", "non-subject", "agent"),

    ("{det1} {person_1} has {verb_ed} a parent. As for the {person_1},", "present perfect", "subject", "agent"),
    ("{det1} {person_1} has been {verb_ed} by a parent. As for the {person_1},", "present perfect", "subject", "patient"),
    ("A parent has {verb_ed} {det1} {person_1}. As for the {person_1},", "present perfect", "non-subject", "patient"),
    ("A parent has been {verb_ed} by {det1} {person_1}. As for the {person_1},", "present perfect", "non-subject", "agent"),

    ("{det1} {person_1} had {verb_ed} a parent. As for the {person_1},", "past perfect", "subject", "agent"),
    ("{det1} {person_1} had been {verb_ed} by a parent. As for the {person_1},", "past perfect", "subject", "patient"),
    ("A parent had {verb_ed} {det1} {person_1}. As for the {person_1},", "past perfect", "non-subject", "patient"),
    ("A parent had been {verb_ed} by {det1} {person_1}. As for the {person_1},", "past perfect", "non-subject", "agent")

]

verbs_ratings = [
    ('acknowledge', '+val+dom'),
    ('intrigue', '+val+dom'),
    ('direct', '+val+dom'),
    ('compliment', '+val+dom'),  # complement?
    ('commend', '+val+dom'),
    ('alarm', '-val+dom'),
    ('intimidate', '-val+dom'),
    ('rush', '-val+dom'),
    ('command', '-val+dom'),
    ('dislike', '-val+dom'),
    ('phone', '+val-dom'),
    ('subdue', '+val-dom'),
    ('surprise', '+val-dom'),
    ('hypnotize', '+val-dom'),
    ('resurrect', '+val-dom'),
    ('agitate', '-val-dom'),
    ('shame', '-val-dom'),
    ('hate', '-val-dom'),
    ('bother', '-val-dom'),
    ('distract', '-val-dom')
]

professions = [
    'colonel',
    'sergeant',
    'soldier',
    'CEO',
    'administrator',
    'scientist',
    'doctor',
    'teacher',
    'manager',
    'solicitor',
    'programmer',
    'midwife',
    'accountant',
    'paralegal',
    'cashier',
    'scrivener',
    'salesperson',
    'bodyguard',
    'agronomist',
    'fisherman',
    'farmer',
    'welder',
    'molder',
    'bookbinder',
    'electrician',
    'tradesperson',
    'chauffeur',
    'housekeeper',
    'farmhand',
    'builder',
    'hawker',
    'binman'
]

DEFAULT_HF_CACHE_DIR = "/mnt/hum01-rds/Nikolaev_Dmitry/dominik-llama/extremism_detection/hf_cache"

def _configure_hf_cache_env(hf_cache_dir: str | None) -> None:
    """
    Configure Hugging Face cache environment variables *before* importing transformers.

    This is important because some `transformers` versions expose `from_pretrained`
    signatures that accept `**kwargs` only, so signature-based detection can miss
    `cache_dir`. Env vars remain a stable way to force a cache location.
    """

    if not hf_cache_dir:
        return
    root = str(Path(hf_cache_dir).expanduser())
    os.environ["HF_HOME"] = root
    os.environ["HF_HUB_CACHE"] = str(Path(root) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(root) / "transformers")
    os.environ["HF_ASSETS_CACHE"] = str(Path(root) / "assets")


def _preferred_hf_token_kwarg(from_pretrained_callable: Any) -> str:
    """
    Transformers/HF Hub have changed auth kwarg names over time.

    Some versions expose `token`, others `use_auth_token`, and some accept only
    `**kwargs` (so `inspect.signature` can't tell us). When we can't detect an
    explicit parameter name, we pick a key by inspecting the source when
    possible.
    """

    try:
        src = inspect.getsource(from_pretrained_callable)
    except Exception:
        return "token"
    return "use_auth_token" if "use_auth_token" in src else "token"


def _select_device(device: str) -> torch.device:
    import torch

    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def _load_hf_token(hf_token: str | None, hf_token_env: str | None) -> str | None:
    if hf_token:
        return hf_token
    if hf_token_env:
        return os.environ.get(hf_token_env)
    return None


def _ensure_pad_token(tokenizer: Any, model: Any) -> None:
    if getattr(tokenizer, "pad_token", None):
        return
    if getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
        return
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))


def _get_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no tokens for {text!r}")
    return token_ids[0]


def _default_output_path(model_tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_tag).strip("._-")
    if not safe:
        safe = "model"
    return str(Path("modelling_data") / f"he_she_odds_results__{safe}.csv")


def _resolve_output_path(requested: str | None, model_tag: str) -> Path:
    default_path = Path(_default_output_path(model_tag))
    if not requested:
        return default_path
    requested_path = Path(requested)
    if requested_path.parent == Path("."):
        return default_path.parent / requested_path.name
    return requested_path


def _load_profession_frequencies(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("profession_counts")
    if not isinstance(counts, dict):
        raise ValueError(f"Expected 'profession_counts' dict in {path}")
    out: dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(key, str) and isinstance(value, int):
            out[key] = value
    return out


def _validate_profession_frequencies(profession_frequencies: dict[str, int]) -> None:
    missing = [profession for profession in professions if profession not in profession_frequencies]
    if missing:
        raise ValueError(f"Missing profession counts for: {', '.join(missing)}")

def generate_sentences() -> list[str]:
    sentences: list[str] = []
    for template, _, _, _ in templates:
        for profession in professions:
            for verb, _ in verbs_ratings:
                sentences.append(format_template(template, profession, verb))
    return sentences


def _hf_from_pretrained_kwargs(
    from_pretrained_callable: Any, resolved_token: str | None, cache_dir: str | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(from_pretrained_callable).parameters
        has_var_keyword = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
    except (TypeError, ValueError):
        params = {}
        has_var_keyword = False
    if resolved_token:
        if "token" in params:
            kwargs["token"] = resolved_token
        elif "use_auth_token" in params:
            kwargs["use_auth_token"] = resolved_token
        elif has_var_keyword:
            kwargs[_preferred_hf_token_kwarg(from_pretrained_callable)] = resolved_token
    if cache_dir and ("cache_dir" in params or has_var_keyword):
        kwargs["cache_dir"] = cache_dir
    return kwargs


def _maybe_add_kwarg(
    callable_obj: Any,
    kwargs: dict[str, Any],
    key: str,
    value: Any,
    allow_var_keyword: bool = False,
) -> dict[str, Any]:
    try:
        params = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    explicit_names = {param.name for param in params}
    if key in explicit_names or (allow_var_keyword and has_var_keyword):
        kwargs[key] = value
    return kwargs


def _tokenizer_trust_kwargs(callable_obj: Any, trust_remote_code: bool) -> dict[str, Any]:
    if not trust_remote_code:
        return {}
    return _maybe_add_kwarg(callable_obj, {}, "trust_remote_code", True, allow_var_keyword=True)


def run(
    model_tag: str,
    hf_token: str | None,
    hf_token_env: str | None,
    device: str,
    hf_cache_dir: str | None,
    progress: bool,
    tokenizer_mode: str,
    trust_remote_code: bool,
) -> pd.DataFrame:
    import pandas as pd
    import torch

    _configure_hf_cache_env(hf_cache_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore

    profession_frequencies = _load_profession_frequencies(Path("profession_counts.json"))
    _validate_profession_frequencies(profession_frequencies)

    resolved_token = _load_hf_token(hf_token, hf_token_env)
    tok_kwargs = _hf_from_pretrained_kwargs(AutoTokenizer.from_pretrained, resolved_token, hf_cache_dir)
    tok_kwargs.update(_tokenizer_trust_kwargs(AutoTokenizer.from_pretrained, trust_remote_code))
    if tokenizer_mode in {"fast", "slow"}:
        tok_kwargs = _maybe_add_kwarg(
            AutoTokenizer.from_pretrained, tok_kwargs, "use_fast", tokenizer_mode == "fast", allow_var_keyword=True
        )
    else:
        tok_kwargs = _maybe_add_kwarg(
            AutoTokenizer.from_pretrained, tok_kwargs, "use_fast", True, allow_var_keyword=True
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_tag, **tok_kwargs)
    except Exception:
        if tokenizer_mode == "fast":
            raise
        tok_kwargs = _hf_from_pretrained_kwargs(AutoTokenizer.from_pretrained, resolved_token, hf_cache_dir)
        tok_kwargs.update(_tokenizer_trust_kwargs(AutoTokenizer.from_pretrained, trust_remote_code))
        tok_kwargs = _maybe_add_kwarg(
            AutoTokenizer.from_pretrained, tok_kwargs, "use_fast", False, allow_var_keyword=True
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_tag, **tok_kwargs)
        except Exception as slow_error:
            if tokenizer_mode == "auto":
                raise RuntimeError(
                    f"Failed to load tokenizer for {model_tag!r}. "
                    "Tried fast tokenizer first, then AutoTokenizer with use_fast=False."
                ) from slow_error
            raise

    model_kwargs = _hf_from_pretrained_kwargs(AutoModelForCausalLM.from_pretrained, resolved_token, hf_cache_dir)
    model_kwargs = _maybe_add_kwarg(
        AutoModelForCausalLM.from_pretrained,
        model_kwargs,
        "trust_remote_code",
        trust_remote_code,
        allow_var_keyword=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_tag, **model_kwargs)
    target_device = _select_device(device)
    model.to(target_device)
    model.eval()

    _ensure_pad_token(tokenizer, model)

    embedding_layer = model.get_input_embeddings()
    if embedding_layer is None or not hasattr(embedding_layer, "weight"):
        raise RuntimeError(f"Model {model_tag!r} does not expose input embeddings.")
    embedding_weight = embedding_layer.weight

    profession_emb_norms: dict[str, float] = {}
    for profession in professions:
        token_ids = tokenizer.encode(f" {profession}", add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode(profession, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Tokenizer produced no tokens for profession {profession!r}")
        token_tensor = torch.tensor(token_ids, device=embedding_weight.device)
        vec = embedding_weight.index_select(0, token_tensor).mean(dim=0)
        profession_emb_norms[profession] = torch.linalg.vector_norm(vec).item()

    he_idx = _get_token_id(tokenizer, " he")
    she_idx = _get_token_id(tokenizer, " she")

    bos = getattr(tokenizer, "bos_token", None) or ""

    rows: list[dict[str, Any]] = []
    total = len(templates) * len(professions) * len(verbs_ratings)
    iterable = (
        (template, tense, syntactic_role, semantic_role, profession, verb, val_dom_str)
        for template, tense, syntactic_role, semantic_role in templates
        for profession in professions
        for verb, val_dom_str in verbs_ratings
    )
    if progress and tqdm is not None:
        iterable = tqdm(iterable, total=total, desc="Scoring", unit="prompt")

    for template, tense, syntactic_role, semantic_role, profession, verb, val_dom_str in iterable:
                valence = val_dom_str[:4]
                dominance = val_dom_str[4:]

                prefix = format_template(template, profession, verb)

                inputs = tokenizer(bos + prefix, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                he_prob = probs[he_idx]
                she_prob = probs[she_idx]

                denom = he_prob + she_prob
                he_scaled = he_prob / denom
                she_scaled = she_prob / denom
                he_odds = he_scaled / she_scaled

                rows.append(
                    {
                        "model_tag": model_tag,
                        "template": template,
                        "tense": tense,
                        "syntactic_role": syntactic_role,
                        "semantic_role": semantic_role,
                        "valence": valence,
                        "dominance": dominance,
                        "verb": verb,
                        "profession": profession,
                        "frequency": profession_frequencies[profession],
                        "lex_emb_norm": profession_emb_norms[profession],
                        "prefix": prefix,
                        "he_prob": he_prob.item(),
                        "she_prob": she_prob.item(),
                        "he_she_odds_ratio": he_odds.item(),
                        "log_he_she_odds": torch.log(he_odds).item(),
                    }
                )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-tag",
        default="gpt2",
        help="Hugging Face model id, e.g. gpt2, meta-llama/Llama-3.2-3B.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write CSV results (default: derived from --model-tag).",
    )
    parser.add_argument(
        "--sentences-path",
        default=None,
        help="Path to write generated sentences (default: derived from --model-tag). "
        "Only written if the file does not already exist.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (optional). Prefer --hf-token-env for local use.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN_GATED",
        help="Environment variable name containing Hugging Face token.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=DEFAULT_HF_CACHE_DIR,
        help="Cache directory for Hugging Face/Transformers downloads.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    parser.add_argument(
        "--tokenizer",
        default="auto",
        choices=["auto", "fast", "slow"],
        help="Tokenizer mode. Use 'slow' to avoid fast-tokenizer JSON incompatibilities on older installs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable Hugging Face 'trust_remote_code' for custom model/tokenizer classes.",
    )

    args = parser.parse_args()
    output_path = _resolve_output_path(args.output, args.model_tag)
    sentences_path = args.sentences_path or "sentences.txt"
    write_verb_forms(Path("modelling_data") / "verb_forms.txt")

    if not os.path.exists(sentences_path):
        sentences = generate_sentences()
        with open(sentences_path, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence)
                f.write("\n")

    df = run(
        model_tag=args.model_tag,
        hf_token=args.hf_token,
        hf_token_env=args.hf_token_env,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        progress=not args.no_progress,
        tokenizer_mode=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
