import json
from pathlib import Path

from .config import EXPECTED_REPS, log


def load_all_results(results_dir: Path):
    results = []

    for jf in sorted(results_dir.glob("rep_*/results_*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Skipping {jf}: {e}")
            continue

        try:
            rep = int(jf.parent.name.replace("rep_", ""))
        except Exception:
            log.warning(f"Skipping invalid rep folder: {jf.parent.name}")
            continue

        if rep not in EXPECTED_REPS:
            continue

        for r in data:
            r["rep"] = rep

        results.extend(data)
        log.info(f"Loaded {len(data)} from {jf}")

    log.info(f"Total results loaded: {len(results)}")
    return results

