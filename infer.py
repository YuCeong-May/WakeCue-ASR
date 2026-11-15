"""WakeCue-ASR inference from the command line.

Transcribe the speaker who says a text cue (usually the wake word):

    python infer.py --audio examples/midea_pos_2465.wav --cue 小美小美

Several files are joined in order with no gap, e.g. a wake-word recording and
the command recording that follows it:

    python infer.py --audio kws.wav cmd.wav --cue 小美小美

Batch mode reads a JSON-lines list with one {"audio": ..., "cue": ...} object
per line; "audio" is a path or a list of paths, relative to the list file:

    python infer.py --list test.jsonl --output result.jsonl

Each output line repeats the input fields and adds "text", "rejected" and "raw".
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from inference import TextCueGuidedTSASR
from inference.inference import SAMPLE_RATE, load_audio

RESULT_KEYS = ("text", "rejected", "raw")


def read_list(path):
    """JSON-lines list -> [(record, audio paths, line reference)]."""
    root = Path(path).parent
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{where}: invalid JSON ({exc})")
            if not isinstance(record, dict):
                sys.exit(f"{where}: expected a JSON object")
            audio = record.get("audio")
            paths = [audio] if isinstance(audio, str) else audio
            if not (isinstance(paths, list) and paths and all(isinstance(p, str) for p in paths)):
                sys.exit(f'{where}: "audio" must be a path or a list of paths')
            if not isinstance(record.get("cue"), str):
                sys.exit(f'{where}: "cue" must be a string')
            if any(key in record for key in RESULT_KEYS):
                sys.exit(f"{where}: {', '.join(RESULT_KEYS)} are output fields; rename them")
            items.append((record, [root / p for p in paths], where))
    return items


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", nargs="+", metavar="WAV", help="audio file(s), joined in order")
    source.add_argument("--list", metavar="JSONL", help='JSON lines with "audio" and "cue"')
    parser.add_argument("--cue", help="text cue for --audio, e.g. the wake word")
    parser.add_argument("--output", metavar="JSONL", help="write the results here, not to stdout")
    parser.add_argument(
        "--weights",
        default="YuCeong-May/WakeCue-ASR",
        help="model repo id, or a local model.pt or its directory (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.audio:
        if args.cue is None:
            parser.error("--cue is required with --audio")
        audio = args.audio[0] if len(args.audio) == 1 else args.audio
        items = [({"audio": audio, "cue": args.cue}, [Path(p) for p in args.audio], "--audio")]
    else:
        if args.cue is not None:
            parser.error('--cue cannot be used with --list; give "cue" on each line')
        items = read_list(args.list)
    # Check every file before the 1.6 GB model is loaded.
    for _, paths, where in items:
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            sys.exit(f"{where}: audio file not found: {', '.join(missing)}")

    asr = TextCueGuidedTSASR.from_pretrained(args.weights).to(args.device)
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for record, paths, where in items:
            try:
                wav = np.concatenate([load_audio(str(p)) for p in paths])
                result = asr.transcribe((wav, SAMPLE_RATE), record["cue"])
            except ValueError as exc:
                sys.exit(f"{where}: {exc}")
            record.update((key, result[key]) for key in RESULT_KEYS)
            print(json.dumps(record, ensure_ascii=False), file=out, flush=True)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
