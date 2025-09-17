#!/usr/bin/env python3

import os
import sys
import gzip
import pickle
import argparse
import importlib
import importlib.util
from typing import Callable, List, Tuple, Dict

import numpy as np
import requests

import config
from model import multi_salience


def load_datalog(path: str):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_datalog(path: str, datalog) -> None:
    with open(path, 'wb') as f:
        pickle.dump(datalog, f, pickle.HIGHEST_PROTOCOL)


def ensure_datalog(path: str) -> str:
    alt = path + '.gz'
    if os.path.exists(path):
        return path
    if os.path.exists(alt):
        return alt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(config.DATALOG_ARCHIVE_URL, timeout=600, stream=True)
    if r.status_code != 200:
        print(f"Failed to download datalog: HTTP {r.status_code}")
        sys.exit(1)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, path)
    return path


def import_generate_func(spec: str) -> Callable[[int], List[List[float]]]:
    if os.path.isfile(spec):
        mod_name = os.path.splitext(os.path.basename(spec))[0]
        module_spec = importlib.util.spec_from_file_location(mod_name, spec)
        module = importlib.util.module_from_spec(module_spec)
        loader = module_spec.loader  # type: ignore
        loader.exec_module(module)  # type: ignore
        return getattr(module, 'generate_embeddings')
    module = importlib.import_module(spec)
    return getattr(module, 'generate_embeddings')


def samples_per_day() -> int:
    return int((24 * 60 * 60) // (12 * int(config.SAMPLE_EVERY)))


def compute_window_sidx(datalog, n_days: int) -> List[int]:
    keys = set()
    for ch in datalog.challenges.values():
        keys.update(ch.sidx.keys())
    ordered = sorted(keys)
    need = samples_per_day() * int(n_days)
    return ordered[-need:] if len(ordered) > need else ordered


def clip_unit(v: float) -> float:
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


def inject_synthetic_embeddings(
    datalog,
    window: List[int],
    generator: Callable[[int], List[List[float]]],
    hotkey: str = 'synthetic_hotkey'
) -> Tuple[int, int]:
    written = 0
    skipped = 0
    sample_every = int(config.SAMPLE_EVERY)
    challenges = config.CHALLENGES
    for sidx in window:
        block = sidx * sample_every
        arrays = generator(block)
        if not isinstance(arrays, list) or len(arrays) != len(challenges):
            skipped += 1
            continue
        for i, spec in enumerate(challenges):
            vec = arrays[i]
            if not isinstance(vec, (list, tuple)) or len(vec) != spec['dim']:
                continue
            clipped = [clip_unit(float(x)) for x in vec]
            if any(x != 0.0 for x in clipped):
                ticker = spec['ticker']
                datalog.challenges[ticker].set_emb(sidx, hotkey, clipped)
                written += 1
    return written, skipped


def run_salience(datalog) -> Tuple[Dict[str, float], List[str]]:
    td = datalog.get_training_data_sync()
    sal = multi_salience(td) if td else {}
    hk = sorted(sal.keys())
    return sal, hk


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generator', required=True, help='Module path or file path providing generate_embeddings(block)')
    p.add_argument('--days', type=int, default=7, help='Number of recent days to inject (default: 7)')
    p.add_argument('--datalog', default=os.path.join(config.STORAGE_DIR, 'mantis_datalog.pkl'))
    args = p.parse_args()

    path = ensure_datalog(args.datalog)
    datalog = load_datalog(path)

    window = compute_window_sidx(datalog, args.days)
    if not window:
        print('No sidx available to inject')
        sys.exit(1)

    gen = import_generate_func(args.generator)
    written, skipped = inject_synthetic_embeddings(datalog, window, gen, 'synthetic_hotkey')

    sal, hotkeys = run_salience(datalog)
    synth_score = sal.get('synthetic_hotkey', 0.0)
    rank = 1 + sorted(((k, v) for k, v in sal.items()), key=lambda kv: kv[1], reverse=True).index(('synthetic_hotkey', synth_score)) if 'synthetic_hotkey' in sal else -1

    print(f"Injected sidx: {len(window)} across {len(config.CHALLENGES)} challenges")
    print(f"Vectors written: {written} | skipped sidx (bad shape): {skipped}")
    print(f"Hotkeys in salience: {len(hotkeys)}")
    if 'synthetic_hotkey' in sal:
        print(f"synthetic_hotkey salience: {synth_score:.6f} | rank {rank} / {len(hotkeys)}")
    else:
        print("synthetic_hotkey not present in salience output (likely filtered by training data)")


if __name__ == '__main__':
    main()


