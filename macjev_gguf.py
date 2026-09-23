"""MacJev GGUF runtime: stock llama.cpp encoder + numpy decision head.

The ModernBERT encoder runs in an unmodified `llama-server` started with
`--embeddings --pooling none` (one hidden-state vector per input token, no BOS/EOS
added for token-id input, no normalization). Token ids come from the shipped HF
tokenizer via macjev_inputs.py (strict budgets, never truncates). The custom decision
head (type embedding, 2 pre-LN transformer layers, marker scorer) ships as
MacJev-322M-4K-Laya-head.safetensors and runs here in numpy float32.

Encoder file: MacJev-322M-4K-Laya-F16.gguf (general.architecture = modern-bert).
APPROXIMATE: stock llama.cpp computes ModernBERT's GeGLU with the tanh
approximation of GELU instead of the exact erf form, so probabilities differ
slightly from the FP32 model (see validation.json). A 32-bit GGUF
is not shipped: on Metal it produced bit-identical results to F16.

Serve the encoder (bind to localhost; the whole <=4096-token sequence must fit one ubatch):
  llama-server -m MacJev-322M-4K-Laya-F16.gguf --host 127.0.0.1 --port 8080 \
      --embeddings --pooling none -c 4096 -b 4096 -ub 4096 -np 1 -fa on
(`python macjev_gguf.py --print-server-command` prints it; `--start-server` launches it.)
MacJevGGUF checks once (GET /props) that the server is serving that file; pass
gguf=None to skip the check.

Dependencies: numpy, tokenizers, safetensors (+ stdlib urllib). No torch/transformers.
This module ranks candidates; it never executes actions. See LICENSE/NOTICE.
"""
import hashlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macjev_inputs import QTYPES, InputBudgetError, TokenizerAdapter, build_sequence  # noqa: E402,F401

HEAD_FILE = "MacJev-322M-4K-Laya-head.safetensors"
MODEL_NAME = "MacJev-322M-4K-Laya"
GGUF_FILES = {"F16": "MacJev-322M-4K-Laya-F16.gguf"}
DEFAULT_GGUF = GGUF_FILES["F16"]
SERVER_FLAGS = ["--embeddings", "--pooling", "none", "-c", "4096", "-b", "4096", "-ub", "4096", "-np", "1", "-fa", "on"]
D_MODEL, N_HEAD, HEAD_DIM, VOCAB = 768, 12, 64, 256000
LN_EPS = 1e-5
_ERF = np.vectorize(math.erf, otypes=[np.float64])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_norm(x, weight, bias, eps=LN_EPS):
    mean = x.mean(-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return (centered / np.sqrt(var + np.float32(eps))) * weight + bias


def _softmax_inplace(scores):
    scores -= scores.max(-1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= scores.sum(-1, keepdims=True, dtype=np.float32)
    return scores


def _gelu_erf(x):
    # exact (erf) GELU, only ever applied to the few marker rows
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + _ERF(x64 / math.sqrt(2.0)))).astype(np.float32)


class NumpyHead:
    """PyTorch DecisionModel head semantics for one unpadded sequence, float32."""

    def __init__(self, tensors, head_layers=2, query_block=1024):
        self.t = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}
        self.head_layers = head_layers
        self.query_block = query_block
        for i in range(head_layers):
            p = f"head.layers.{i}."
            if self.t[p + "self_attn.in_proj_weight"].shape != (3 * D_MODEL, D_MODEL):
                raise ValueError("Unexpected head attention shape")
            # pre-transposed weights for x @ W.T
            for name in ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight"):
                self.t[p + name + ".T"] = np.ascontiguousarray(self.t[p + name].T)
        for name in ("scorer.1.weight", "scorer.3.weight"):
            self.t[name + ".T"] = np.ascontiguousarray(self.t[name].T)

    def _attention(self, q, k, v):
        # q: (H, m, 64) already scaled; k, v: (H, n, 64); full bidirectional attention
        out = np.empty((q.shape[1], D_MODEL), dtype=np.float32)
        for start in range(0, q.shape[1], self.query_block):
            stop = min(start + self.query_block, q.shape[1])
            probs = _softmax_inplace(np.matmul(q[:, start:stop], k.transpose(0, 2, 1)))
            out[start:stop] = np.matmul(probs, v).transpose(1, 0, 2).reshape(stop - start, D_MODEL)
        return out

    def _layer(self, h, i, rows=None):
        """One nn.TransformerEncoderLayer(norm_first=True, relu, eval). If `rows` is given, only those
        query rows are computed/returned (keys/values still come from every token)."""
        t, p = self.t, f"head.layers.{i}."
        n = h.shape[0]
        x = _layer_norm(h, t[p + "norm1.weight"], t[p + "norm1.bias"])
        w, b = t[p + "self_attn.in_proj_weight.T"], t[p + "self_attn.in_proj_bias"]
        k = (x @ w[:, D_MODEL:2 * D_MODEL] + b[D_MODEL:2 * D_MODEL]).reshape(n, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
        v = (x @ w[:, 2 * D_MODEL:] + b[2 * D_MODEL:]).reshape(n, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
        hq, xq = (h, x) if rows is None else (h[rows], x[rows])
        q = (xq @ w[:, :D_MODEL] + b[:D_MODEL]) * np.float32(1.0 / math.sqrt(HEAD_DIM))
        q = q.reshape(-1, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
        attn = self._attention(np.ascontiguousarray(q), np.ascontiguousarray(k), np.ascontiguousarray(v))
        hq = hq + (attn @ t[p + "self_attn.out_proj.weight.T"] + t[p + "self_attn.out_proj.bias"])
        x2 = _layer_norm(hq, t[p + "norm2.weight"], t[p + "norm2.bias"])
        ff = np.maximum(x2 @ t[p + "linear1.weight.T"] + t[p + "linear1.bias"], np.float32(0))
        return hq + (ff @ t[p + "linear2.weight.T"] + t[p + "linear2.bias"])

    def logits(self, hidden, markers, qtype_index):
        t = self.t
        h = np.asarray(hidden, dtype=np.float32) + t["type_emb.weight"][qtype_index]
        markers = np.asarray(markers, dtype=np.int64)
        for i in range(self.head_layers):
            last = i == self.head_layers - 1
            # the scorer only reads marker rows, so the last layer computes only those query rows
            h = self._layer(h, i, rows=markers if last else None)
        z = h if self.head_layers else h[markers]
        z = _layer_norm(z, t["scorer.0.weight"], t["scorer.0.bias"])
        z = _gelu_erf(z @ t["scorer.1.weight.T"] + t["scorer.1.bias"])
        return (z @ t["scorer.3.weight.T"] + t["scorer.3.bias"])[:, 0]


def llama_server_command(model_dir, gguf=DEFAULT_GGUF, host="127.0.0.1", port=8080, binary="llama-server"):
    """argv for the stock llama-server serving the MacJev encoder GGUF."""
    return [binary, "-m", str(Path(model_dir) / gguf), "--host", host, "--port", str(port), *SERVER_FLAGS]


def served_gguf(server_url, timeout=30):
    """Basename of the GGUF the llama-server at server_url is serving (GET /props)."""
    with urllib.request.urlopen(server_url.rstrip("/") + "/props", timeout=timeout) as response:
        return Path(json.loads(response.read()).get("model_path", "")).name


def encoder_hidden_states(server_url, ids, timeout=600):
    """POST token ids to llama-server /embedding; returns (len(ids), 768) float32 per-token states."""
    body = json.dumps({"content": [int(i) for i in ids]}).encode()
    request = urllib.request.Request(server_url.rstrip("/") + "/embedding", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    if not isinstance(data, list) or len(data) != 1 or "embedding" not in data[0]:
        raise RuntimeError("Unexpected llama-server /embedding response; start it with --embeddings --pooling none")
    hidden = np.asarray(data[0]["embedding"], dtype=np.float32)
    if hidden.shape != (len(ids), D_MODEL):
        raise RuntimeError(f"llama-server returned {hidden.shape} for {len(ids)} tokens; "
                           "it must run with --pooling none and -b/-ub >= the sequence length")
    return hidden


class MacJevGGUF:
    def __init__(self, model_dir, server_url="http://127.0.0.1:8080", verify=True, timeout=600, gguf=DEFAULT_GGUF):
        """gguf: encoder file the server must be serving (checked once via GET /props before the first
        request); None skips the check."""
        self.directory = Path(model_dir).resolve(strict=True)
        self.server_url = server_url
        self.timeout = timeout
        if gguf is not None and gguf not in GGUF_FILES.values():
            raise ValueError(f"gguf must be one of {sorted(GGUF_FILES.values())} or None")
        self.gguf = gguf
        self._server_checked = gguf is None
        if verify:
            self._verify_files()
        with safe_open(self.directory / HEAD_FILE, "np") as stream:
            self.config = json.loads(stream.metadata()["macjev.config"])
            tensors = {key: stream.get_tensor(key) for key in stream.keys()}
        agent = json.loads((self.directory / "rl_agent_config.json").read_text())
        self.max_len, self.head_max_len = self.config["max_len"], self.config["head_max_len"]
        if (self.max_len, self.head_max_len) != (4096, 1024):
            raise ValueError("Unexpected model input budget")
        self.temperatures = self.config["temperatures"]
        if set(self.temperatures) != set(QTYPES) or any(not math.isfinite(v) or v <= 0 for v in self.temperatures.values()):
            raise ValueError("Invalid calibration temperatures")
        if agent["temperature"] != [self.temperatures[k] for k in QTYPES] or agent["temperature"] != self.config["temperature"]:
            raise ValueError("rl_agent_config.json and head temperatures differ")
        expected = {"type_emb.weight", "temperature", "scorer.0.weight", "scorer.0.bias", "scorer.1.weight",
                    "scorer.1.bias", "scorer.3.weight", "scorer.3.bias", "act_head.0.weight", "act_head.0.bias",
                    "act_head.2.weight", "act_head.2.bias"}
        expected |= {f"head.layers.{i}.{n}" for i in range(self.config["head_layers"]) for n in (
            "self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")}
        if set(tensors) != expected or any(v.dtype != np.float32 for v in tensors.values()):
            raise ValueError("Head tensor set/dtype mismatch")
        # NOTE: the checkpoint's `temperature` buffer is kept bit-exact but is [1, 1, 1]; calibration uses
        # config["temperatures"] (same values as rl_agent_config.json "temperature"), exactly like macjev.py.
        self.head = NumpyHead(tensors, self.config["head_layers"])
        self.tokenizer = TokenizerAdapter((self.directory / "tokenizer/tokenizer.json").read_text(),
                                          json.loads((self.directory / "tokenizer/tokenizer_config.json").read_text()))

    def _verify_files(self):
        """Check the files this runtime loads against manifest.json (symlinks such as the
        Hugging Face cache are followed; manifest paths must stay inside the package)."""
        listed = json.loads((self.directory / "manifest.json").read_text())["files_sha256"]
        for relative in (HEAD_FILE, "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "rl_agent_config.json"):
            target = self.directory / relative
            if relative not in listed or not target.is_file() or _sha256(target) != listed[relative]:
                raise ValueError("Model package integrity mismatch: " + relative)

    def logits_from_ids(self, ids, markers, qtype_index):
        """Uncalibrated candidate logits (float32) for one complete, unpadded sequence."""
        ids = [int(i) for i in ids]
        markers = [int(m) for m in markers]
        if not ids or len(ids) > self.max_len:
            raise InputBudgetError(f"Sequence length {len(ids)} outside 1..{self.max_len}; nothing is truncated")
        if min(ids) < 0 or max(ids) >= VOCAB:
            raise ValueError("Token id out of vocabulary range")
        if not markers or min(markers) < 0 or max(markers) >= len(ids):
            raise ValueError("Marker positions must index into the sequence")
        if int(qtype_index) not in QTYPES.values():
            raise ValueError("qtype_index must be 0 (choice), 1 (score) or 2 (noul)")
        if not self._server_checked:
            serving = served_gguf(self.server_url)
            if serving != self.gguf:
                raise RuntimeError(f"llama-server at {self.server_url} serves {serving!r}, expected {self.gguf!r} "
                                   "(pass gguf=... to MacJevGGUF to use another tier, or gguf=None to skip this check)")
            self._server_checked = True
        hidden = encoder_hidden_states(self.server_url, ids, self.timeout)
        return self.head.logits(hidden, markers, int(qtype_index))

    def decide(self, state, question):
        started = time.monotonic()
        ids, markers = build_sequence(self.tokenizer, state, question, self.max_len, self.head_max_len)
        logits = self.logits_from_ids(ids, markers, QTYPES[question["t"]]).astype(np.float64)
        scaled = logits / self.temperatures[question["t"]]
        scaled = np.exp(scaled - scaled.max())
        probs = (scaled / scaled.sum()).tolist()
        names = (list(question["crit"]) if question["t"] == "choice" else
                 [str(i) for i in range(len(question["crit"]))] if question["t"] == "score" else ["false", "true"])
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"answer": names[best], "probabilities": dict(zip(names, probs)), "top_probability": probs[best],
                "input_tokens": len(ids), "latency_ms": (time.monotonic() - started) * 1000,
                "model": MODEL_NAME, "actions_executed": False, "backend": "llama.cpp-server"}


if __name__ == "__main__":
    import argparse
    import os
    import signal
    import subprocess
    import urllib.parse
    parser = argparse.ArgumentParser(description="One JSON object per line on stdin: {\"state\": ..., \"question\": ...}. "
                                                 "Needs a stock llama-server serving MacJev-322M-4K-Laya-F16.gguf with "
                                                 + " ".join(SERVER_FLAGS) + " (see --print-server-command / --start-server). "
                                                 "No Mac actions are executed.")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--gguf", default=DEFAULT_GGUF, choices=sorted(GGUF_FILES.values()),
                        help="encoder GGUF the server serves (default: %(default)s)")
    parser.add_argument("--print-server-command", action="store_true", help="print the llama-server command and exit")
    parser.add_argument("--start-server", action="store_true", help="start stock llama-server for --gguf on --server's port; stop it on exit")
    parser.add_argument("--llama-server", default="llama-server", help="llama-server binary for --start-server")
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.server)
    command = llama_server_command(args.model, args.gguf, url.hostname or "127.0.0.1", url.port or 8080, args.llama_server)
    if args.print_server_command:
        print(" ".join(command))
        sys.exit(0)
    process = None
    try:
        if args.start_server:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise SystemExit(f"llama-server exited with code {process.returncode}: {' '.join(command)}")
                try:
                    with urllib.request.urlopen(args.server.rstrip("/") + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise SystemExit("llama-server did not become healthy")
                time.sleep(0.3)
        model = MacJevGGUF(args.model, args.server, gguf=args.gguf)
        for line in sys.stdin:
            if line.strip():
                request = json.loads(line)
                print(json.dumps(model.decide(request["state"], request["question"]), ensure_ascii=False), flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
