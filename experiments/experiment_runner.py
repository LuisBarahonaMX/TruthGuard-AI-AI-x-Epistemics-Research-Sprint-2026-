"""TruthGuard AI 7-model benchmark runner.

Safe defaults: --dry-run, --pilot for small live verification.
Live full run: --live --all. JSONL is append-only, one record per finished case.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_client import MODELS, ModelClient, ResponseFailure


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_configuration():
    benchmark_path = ROOT / "data" / "benchmark_v1.json"
    protocol_path = ROOT / "protocol" / "protocol_v1.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8-sig"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    cases = benchmark["cases"]
    conditions = protocol["conditions"]
    assert len(cases) == 20 and len({c["id"] for c in cases}) == 20
    assert {x for x in "ABC" if conditions[x]["enabled"]} == {"A", "B", "C"}
    assert not conditions["D"]["enabled"], "Condition D requires a separate protocol"
    return cases, conditions, {
        "benchmark_sha256": sha256(benchmark_path),
        "protocol_sha256": sha256(protocol_path),
        "benchmark_version": benchmark.get("version"),
        "protocol_version": protocol.get("version"),
    }


def calculate_conditions(cases, conditions, models):
    return [(m, c, k) for m in models for c in cases for k in ("A", "B", "C") if conditions[k]["enabled"]]


def followup(case, condition, conditions):
    spec = conditions[condition]
    if "follow_up" in spec:
        return spec["follow_up"]
    return spec["follow_up_template"].format(incorrect=case["incorrect"])


def load_existing(path: Path, manifest: dict):
    """Only skip completed records matching benchmark/protocol checksums."""
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Corrupt JSONL at line {n}: {exc}; do not delete evidence") from exc
            if any(record.get(k) != manifest[k] for k in ("benchmark_sha256", "protocol_sha256")):
                raise RuntimeError("Existing output has a different dataset/protocol hash. Choose another --output path")
            if record.get("status") == "completed":
                done.add((record["model"], record["case_id"], record["condition"]))
    return done


def record_cost(record):
    return sum((item.get("estimated_credits_from_usage") or 0) for item in record.get("responses", []))


async def run_one(client, model, case, condition, conditions, manifest):
    prompt = case["question"]
    challenge = followup(case, condition, conditions)
    record = {
        **manifest, "timestamp_utc": utc_now(), "model": model,
        "case_id": case["id"], "category": case["category"],
        "condition": condition, "expected": case["expected"],
        "incorrect": case["incorrect"], "question": prompt,
        "follow_up": challenge, "status": "started", "responses": [],
        "error": None,
    }
    try:
        first = await client.generate(model, [{"role": "user", "content": prompt}])
        record["responses"].append(first)
        second = await client.generate(model, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": first["text"]},
            {"role": "user", "content": challenge},
        ])
        record["responses"].append(second)
        record["status"] = "completed"
    except ResponseFailure as exc:
        if exc.response_data:
            record["responses"].append(exc.response_data)
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["estimated_credits_from_usage"] = record_cost(record)
    record["finished_at_utc"] = utc_now()
    return record


def write_jsonl_sync(path, record):
    # Single writer on event loop; each complete JSON line is flushed and fsynced.
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def export_summary(path):
    if not path.exists():
        return
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            # Only the last attempt per key appears in summary. Raw JSONL never rewritten.
            rows.append(r)
    latest = {(r["model"], r["case_id"], r["condition"]): r for r in rows}
    summary_path = path.with_suffix(".summary.csv")
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "case_id", "condition", "status", "initial_text", "final_text", "input_tokens", "output_tokens", "estimated_credits_from_usage", "error"])
        writer.writeheader()
        for _, r in sorted(latest.items()):
            usage = [a.get("usage", {}) for a in r["responses"]]
            writer.writerow({
                "model": r["model"], "case_id": r["case_id"], "condition": r["condition"],
                "status": r["status"],
                "initial_text": r["responses"][0].get("text", "") if len(r["responses"]) >= 1 else "",
                "final_text": r["responses"][1].get("text", "") if len(r["responses"]) >= 2 else "",
                "input_tokens": sum(int(u.get("input_tokens", 0)) for u in usage),
                "output_tokens": sum(int(u.get("output_tokens", 0)) for u in usage),
                "estimated_credits_from_usage": r["estimated_credits_from_usage"],
                "error": json.dumps(r.get("error"), ensure_ascii=False),
            })
    print("Summary CSV:", summary_path)
    print("Accumulated estimate of usage (including attempted records):", round(sum(r.get("estimated_credits_from_usage", 0) for r in rows), 8), "credits")
    print("NOTE: price * observed tokens is NOT exact invoiced spend; see Adaption Billing.")


async def execute(args, jobs, conditions, manifest, output):
    done = load_existing(output, manifest)
    todo = [(m,c,k) for m,c,k in jobs if (m,c["id"],k) not in done]
    print(f"Jobs: {len(jobs)} total, {len(done)} previously completed, {len(todo)} pending")
    if not todo:
        export_summary(output)
        return
    queue = asyncio.Queue()
    for job in todo:
        queue.put_nowait(job)
    counts = {"completed": 0, "failed": 0}

    async with ModelClient(timeout=args.timeout, max_output_tokens=args.max_output_tokens, retries=args.retries) as client:
        async def worker(worker_id):
            while True:
                try:
                    model, case, condition = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await run_one(client, model, case, condition, conditions, manifest)
                    write_jsonl_sync(output, record)
                    counts[record["status"]] += 1
                    print(f"[{counts['completed'] + counts['failed']}/{len(todo)}] {model} {case['id']} {condition} {record['status']} ~{record['estimated_credits_from_usage']:.6f} credits", flush=True)
                finally:
                    queue.task_done()
        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
    print("Finished:", counts)
    export_summary(output)


def main():
    parser = argparse.ArgumentParser(description="TruthGuard AI seven-model experiment runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print plan, no API calls")
    mode.add_argument("--live", action="store_true", help="Allow real API calls")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--pilot", action="store_true", help="MATH-001 across 3 conditions and selected models")
    scope.add_argument("--all", action="store_true", help="All 20 benchmark cases")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "seven_models_v1.jsonl")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 7:
        parser.error("--concurrency must be between 1 and 7")
    if not args.pilot and not args.all:
        parser.error("Choose --pilot or --all")
    if args.max_output_tokens <= 0 or args.retries < 0:
        parser.error("max output must be positive, retries nonnegative")
    cases, conditions, manifest = load_configuration()
    if args.pilot:
        cases = [case for case in cases if case["id"] == "MATH-001"]
    jobs = calculate_conditions(cases, conditions, args.models)
    output = args.output.resolve()
    print("Models:", ", ".join(args.models))
    print("Dataset SHA256:", manifest["benchmark_sha256"])
    print("Protocol SHA256:", manifest["protocol_sha256"])
    print("Conversations:", len(jobs), "| API calls expected:", 2 * len(jobs))
    print("Output:", output)
    if args.dry_run:
        print("DRY RUN: no API calls and no output files written")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(execute(args, jobs, conditions, manifest, output))


if __name__ == "__main__":
    main()
