"""STEP 6 - review examples in the terminal (resumable). YOU decide what the model is trained on.

Keys: a = approve as is | e = edit the answer | r = reject | s = skip for now | q = quit
Run:  python -m src.slm.data.review [--only-flagged]
"""
from __future__ import annotations

import argparse

from src.slm.data.common import PATHS, read_jsonl, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-flagged", action="store_true")
    args = ap.parse_args()
    inputs = {i["qid"]: i for i in read_jsonl(PATHS["inputs"])}
    rows = list(read_jsonl(PATHS["review"]))
    todo = [r for r in rows if r["status"] == "pending" and (r["auto_flags"] or not args.only_flagged)]
    print(f"{len(todo)} examples to review")
    for n, r in enumerate(todo, 1):
        item = inputs[r["qid"]]
        print(f"\n===== {n}/{len(todo)}  [{r['label']}/{r['variant']}]  flags: {r['auto_flags'] or 'none'}")
        for i, e in enumerate(item["evidence"], 1):
            print(f"[{i}] {e['title']} ({e['source_url']})\n    {' '.join(e['text'].split())[:350]}")
        print(f"\nQUESTION: {r['question']}\nDRAFT:    {r['draft']}")
        while True:
            k = input("[a]pprove [e]dit [r]eject [s]kip [q]uit > ").strip().lower()
            if k in ("a", "e", "r", "s", "q"):
                break
        if k == "q":
            break
        if k == "a":
            r["status"], r["final_answer"] = "approved", r["draft"]
        elif k == "e":
            r["final_answer"], r["status"] = input("new answer: ").strip(), "edited"
        elif k == "r":
            r["status"] = "rejected"
        write_jsonl(PATHS["review"], rows)  # save after every decision
    write_jsonl(PATHS["review"], rows)
    print("saved", PATHS["review"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
