"""Add a task-vocabulary LM head (output_sub.*, output_sub_map.weight) to a Qwen3.5 GGUF.

The expiration-date model only ever emits
    "The expiration date on the package reads YYYY-MM-DD." | "No expiration date is visible on this package."
    + "\n" + {"year": .., "month": .., "day": ..}
(fine-tuning targets, see finetune_groupA.make_target). We tokenize every such answer (years 1990-2099 or
????, months, days or ??, null variants) and add the end-of-turn/think tokens and every single-byte /
single-printable-character token as a fallback. Rows of the tied output matrix for those tokens are
stored as F16; all other tokens map to a zero pad row whose bias is -inf.

  python add_task_vocab.py in.gguf out.gguf
"""
import json
import sys

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, "/home/user/llama.cpp/gguf-py")
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

TOK = Tokenizer.from_file("/home/user/qwen/groupA_a3_merged/tokenizer.json")


def answers():
    years = [str(y) for y in range(1990, 2100)] + [None]
    for y in years:
        for m in list(range(1, 13)) + [None]:
            for d in list(range(1, 32)) + [None]:
                if y is None and m is None and d is None:
                    s = "No expiration date is visible on this package."
                else:
                    s = (f"The expiration date on the package reads {y if y else '????'}-"
                         f"{f'{m:02d}' if m else '??'}-{f'{d:02d}' if d else '??'}.")
                js = json.dumps({"year": int(y) if y else None, "month": m, "day": d})
                yield f"{s}\n{js}"


def task_vocab(n_vocab, eog_ids):
    ids = set(eog_ids)
    for enc in TOK.encode_batch(list(answers()), add_special_tokens=False):
        ids.update(enc.ids)
    for t in ("<|im_end|>", "<|endoftext|>", "<think>", "</think>", "\n", "\n\n", "```", "json"):
        ids.update(TOK.encode(t, add_special_tokens=False).ids)
    vocab = TOK.get_vocab()
    for tok, i in vocab.items():   # single byte-level characters: lets any text still be spelled out
        if i < n_vocab and len(tok) == 1:
            ids.add(i)
    return sorted(i for i in ids if i < n_vocab)


def main(src, dst):
    r = gguf.GGUFReader(src)
    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    emb = next(t for t in r.tensors if t.name == "output.weight") if any(t.name == "output.weight" for t in r.tensors) \
        else next(t for t in r.tensors if t.name == "token_embd.weight")
    n_embd, n_vocab = int(emb.shape[0]), int(emb.shape[1])
    eog = []
    for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if k in r.fields:
            eog.append(int(r.fields[k].contents()))
    ids = task_vocab(n_vocab, eog)
    W = dequantize(emb.data, emb.tensor_type).reshape(n_vocab, n_embd)
    sub = np.zeros((len(ids) + 1, n_embd), dtype=np.float16)
    sub[:-1] = W[ids].astype(np.float16)
    bias = np.zeros(len(ids) + 1, dtype=np.float32)
    bias[-1] = -np.inf
    mp = np.full(n_vocab, len(ids), dtype=np.int32)
    mp[ids] = np.arange(len(ids), dtype=np.int32)
    print(f"{src}: n_vocab {n_vocab}, task vocabulary {len(ids)} tokens "
          f"({sub.nbytes / 2**20:.2f} MiB head vs {emb.n_bytes / 2**20:.1f} MiB full)")

    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    new = [("output_sub.weight", sub), ("output_sub.bias", bias), ("output_sub_map.weight", mp)]
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, a in new:
        w.add_tensor_info(name, a.shape, a.dtype, a.nbytes)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    for _, a in new:
        w.write_tensor_data(a)
    w.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
