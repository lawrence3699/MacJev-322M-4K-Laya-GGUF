"""Shared MacJev input format without PyTorch/Transformers dependencies.

Derived from the Apache-2.0 Laya input representation; see NOTICE/LICENSE.
"""
import hashlib
import json

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InputBudgetError(ValueError): pass


class TokenizerAdapter:
    def __init__(self, serialized_tokenizer, config):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_str(serialized_tokenizer)
        self.backend.no_padding(); self.backend.no_truncation()
        for name in ("mask", "cls", "sep", "pad"):
            value = config[name + "_token"]
            token = value if isinstance(value, str) else value["content"]
            index = self.backend.token_to_id(token)
            if index is None: raise ValueError("Tokenizer special token is absent")
            setattr(self, name + "_token", token); setattr(self, name + "_token_id", index)
    def __call__(self, values, **kwargs):
        return {"input_ids": [row.ids for row in self.backend.encode_batch(values, add_special_tokens=False)]}


def criterion(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def options(q):
    if q.get("t") not in QTYPES or not isinstance(q.get("ins"), str): raise ValueError("Invalid question type/instruction")
    crit = q.get("crit")
    if q["t"] == "choice":
        if not isinstance(crit, dict) or not crit or not all(isinstance(k,str) for k in crit): raise ValueError("Choice needs an ordered nonempty mapping")
        return [k if v is None or v == "" else f"{k}: {criterion(v)}" for k,v in crit.items()]
    if q["t"] == "score":
        if not isinstance(crit, list) or not crit: raise ValueError("Score needs ordered criteria")
        return [f"level {i}: {criterion(v)}" for i,v in enumerate(crit)]
    crit = crit or {}
    if not isinstance(crit, dict): raise ValueError("Noul criteria must be a mapping")
    return ["false: " + (criterion(crit.get("false")) if crit.get("false") not in (None, "") else "no, the statement does not hold"),
            "true: " + (criterion(crit.get("true")) if crit.get("true") not in (None, "") else "yes, the statement holds")]


def build_sequence(tokenizer, state, q, max_len=4096, head_max_len=1024):
    clean = lambda value: str(value).replace(tokenizer.mask_token, " ")
    text = state if isinstance(state,str) else json.dumps(state, ensure_ascii=False)
    encoded = tokenizer([f'{q["t"]} question: {clean(q["ins"])}', *[" " + clean(v) for v in options(q)], clean(text)])["input_ids"]
    pieces = [[tokenizer.mask_token_id] + row for row in encoded[1:-1]]
    if len(encoded[0]) + sum(map(len,pieces)) > head_max_len: raise InputBudgetError("Complete question/options exceed the 1024-token budget")
    ids = [tokenizer.cls_token_id] + encoded[0] + [tokenizer.sep_token_id]; markers = []
    for piece in pieces: markers.append(len(ids)); ids.extend(piece)
    ids += [tokenizer.sep_token_id] + encoded[-1] + [tokenizer.sep_token_id]
    if len(ids) > max_len: raise InputBudgetError("Complete input exceeds 4096 tokens; no truncation applied")
    return ids, markers
