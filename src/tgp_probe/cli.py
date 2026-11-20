"""Command line: ``tgp-probe {score,train,serve,calibrators}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _read(value: Optional[str], file: Optional[str], name: str) -> str:
    if file:
        return sys.stdin.read() if file == "-" else Path(file).read_text(encoding="utf-8")
    if value is None:
        raise SystemExit(f"error: pass --{name} or --{name}-file")
    return value


def _add_model_args(p: argparse.ArgumentParser, calibrator_default: Optional[str] = "auto") -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"evaluator model (default: {DEFAULT_MODEL})")
    p.add_argument("--calibrator", default=calibrator_default,
                   help="calibrator JSON path, 'auto' (bundled for --model) or 'none'")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detected)")
    p.add_argument("--max-context-tokens", type=int, default=None)
    p.add_argument("--max-response-tokens", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true")


def _load_probe(args, calibrator=None):
    from .probe import TGPProbe

    cal = args.calibrator if calibrator is None else calibrator
    if isinstance(cal, str) and cal.lower() == "none":
        cal = None
    return TGPProbe.from_pretrained(
        args.model, calibrator=cal, device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_context_tokens=args.max_context_tokens,
        max_response_tokens=args.max_response_tokens,
    )


def cmd_score(args) -> int:
    probe = _load_probe(args)
    context = _read(args.context, args.context_file, "context")
    response = _read(args.response, args.response_file, "response")
    result = probe.score(context, response, return_tokens=args.tokens)
    print(json.dumps(result.to_dict(), indent=2))
    if result.score is None:
        print("note: no calibrator loaded, so only raw features were returned "
              "(train one with `tgp-probe train`).", file=sys.stderr)
    return 0


def cmd_train(args) -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from .calibrator import Calibrator
    from .data import LOADERS
    from .features import features_to_vector, resolve_feature_set

    probe = _load_probe(args, calibrator="none")
    names = resolve_feature_set(args.features)

    X_tr, y_tr, test_sets = [], [], {}
    for ds in args.datasets:
        print(f"[{ds}] loading {args.n} pairs ...", file=sys.stderr)
        pairs = LOADERS[ds](args.n)
        ctx, resp, y = zip(*pairs)
        feats = probe.extract_batch(ctx, resp, batch_size=args.batch_size, show_progress=True)
        X = np.array([features_to_vector(f, names) for f in feats])
        y = np.array(y)
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
        X_tr.append(Xa); y_tr.append(ya); test_sets[ds] = (Xb, yb)

    meta = {
        "model": args.model,
        "dtype": str(next(probe.model.parameters()).dtype),
        "feature_set": args.features,
        "datasets": args.datasets,
        "n_per_dataset": args.n,
        "max_context_tokens": probe.max_context_tokens,
        "max_response_tokens": probe.max_response_tokens,
    }
    X_all, y_all = np.vstack(X_tr), np.concatenate(y_tr)
    meta["n_train"] = int(len(y_all))
    cal = Calibrator.fit(X_all, y_all, names, C=args.C, metadata=meta)

    print(f"\nHeld-out AUROC ({int(args.test_size * 100)}% split per dataset)", file=sys.stderr)
    aurocs = {}
    for ds, (Xb, yb) in test_sets.items():
        aurocs[ds] = round(float(roc_auc_score(yb, cal.predict_proba(Xb))), 4)
        print(f"  {ds:<24} {aurocs[ds]:.3f}  (n={len(yb)})", file=sys.stderr)
    cal.metadata["test_auroc"] = aurocs

    out = cal.save(args.out)
    print(f"\nSaved calibrator -> {out}", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("serve needs uvicorn: pip install 'tgp-probe[server]'")
    from .server import create_app

    probe = _load_probe(args)
    if probe.calibrator is None:
        print("warning: serving without a calibrator, so /score returns features only.", file=sys.stderr)
    uvicorn.run(create_app(probe), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_calibrators(args) -> int:
    from .calibrator import list_bundled

    names = list_bundled()
    print("\n".join(names) if names else "(no bundled calibrators)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    from .data import LOADERS
    from .features import FEATURE_SETS

    parser = argparse.ArgumentParser(prog="tgp-probe", description="Trajectory Geometry Probe")
    parser.add_argument("--version", action="version", version=f"tgp-probe {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score", help="score one (context, response) pair, printed as JSON")
    _add_model_args(p)
    p.add_argument("--context"); p.add_argument("--context-file", help="path, or '-' for stdin")
    p.add_argument("--response"); p.add_argument("--response-file", help="path, or '-' for stdin")
    p.add_argument("--tokens", action="store_true", help="include per-token traces")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("train", help="fit a calibrator on labelled benchmarks")
    _add_model_args(p, calibrator_default="none")
    p.add_argument("--datasets", nargs="+", choices=sorted(LOADERS),
                   default=["halueval_qa", "halueval_summarization", "halueval_dialogue"])
    p.add_argument("--n", type=int, default=500, help="balanced pairs per dataset")
    p.add_argument("--features", choices=sorted(FEATURE_SETS), default="paper")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", required=True, help="where to write the calibrator JSON")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="run the HTTP API")
    _add_model_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("calibrators", help="list calibrators bundled with the package")
    p.set_defaults(func=cmd_calibrators)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
