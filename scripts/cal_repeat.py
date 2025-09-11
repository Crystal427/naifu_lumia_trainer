#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def scan_dataset(root: Path) -> Dict[str, int]:
    artist_to_count: Dict[str, int] = {}
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        artist_name = item.name
        count = 0
        # Count only image files in the top level of artist folder (non-recursive).
        for f in item.iterdir():
            if is_image_file(f):
                count += 1
        if count > 0:
            artist_to_count[artist_name] = count
    if not artist_to_count:
        raise RuntimeError("No artist folders with images were found.")
    return artist_to_count


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_auto_target_effective(
    counts: List[int],
    cap: int,
    max_repeat: int,
) -> int:
    """
    Heuristic for auto target:
    - Ignore artists with counts > cap for target computation (they will use repeat=1 anyway).
    - Start from the 75th percentile-ish via mean of top-half to avoid too many repeats near 10.
    - Ensure target doesn't force the smallest artist to hit max_repeat; keep a small margin.
    """
    capped = [c for c in counts if c > 0 and c <= cap]
    if not capped:
        # Fallback if all are > cap
        return cap

    sorted_c = sorted(capped)
    mid = len(sorted_c) // 2
    upper_half = sorted_c[mid:] if mid > 0 else sorted_c
    base = int(mean(upper_half)) if upper_half else int(mean(sorted_c))
    base = max(1, min(base, cap))

    min_c = sorted_c[0]
    # Keep a margin so smallest artist does not map to exactly max_repeat
    max_target_allowed = int(math.floor((max_repeat - 0.5) * max(1, min_c)))
    target = min(base, cap, max_target_allowed)
    target = max(target, int(mean(sorted_c)))  # avoid too small target
    target = max(2, target)  # avoid degenerate 1
    return target


def compute_repeats(
    artist_to_count: Dict[str, int],
    cap: int = 1500,
    max_repeat: int = 10,
    target_effective: int = None,
    avoid_exact_extremes: bool = True,
) -> Tuple[Dict[str, int], int]:
    """
    Returns:
        repeats: mapping artist -> repeat (int)
        target_effective: the target effective sample count used
    """
    counts = list(artist_to_count.values())
    if target_effective is None:
        target_effective = compute_auto_target_effective(counts, cap, max_repeat)

    repeats: Dict[str, int] = {}
    for artist, c in artist_to_count.items():
        # Large classes: force 1
        if c > cap:
            r = 1
        else:
            # Real-valued base to reach target
            r_float = target_effective / max(1, c)
            r_float = clamp(r_float, 1.0, float(max_repeat))

            # Integer rounding
            r = int(round(r_float))

            # Avoid exact extremes if possible
            if avoid_exact_extremes:
                if r >= max_repeat:
                    r = max_repeat - 1
                if r <= 1 and c <= int(0.9 * cap):
                    # Only bump off 1 when not a truly large class
                    r = 2

            # Respect the >cap rule strictly
            if c > cap:
                r = 1

            # Final clamp
            r = max(1, min(max_repeat, r))

        repeats[artist] = r

    return repeats, target_effective


def summarize_effective(artist_to_count: Dict[str, int], repeats: Dict[str, int]) -> Tuple[float, float]:
    effective_counts = [artist_to_count[a] * repeats[a] for a in artist_to_count.keys()]
    mu = mean(effective_counts)
    sigma = pstdev(effective_counts) if len(effective_counts) > 1 else 0.0
    cv = (sigma / mu) if mu > 0 else 0.0
    return mu, cv


def save_csv(out_path: Path, artist_to_count: Dict[str, int], repeats: Dict[str, int]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "count", "repeat", "effective_count"])
        for artist in sorted(artist_to_count.keys()):
            c = artist_to_count[artist]
            r = repeats[artist]
            writer.writerow([artist, c, r, c * r])


def save_txt(out_path: Path, repeats: Dict[str, int]) -> None:
    """
    Save repeats in bullet list format:
    - [artist, repeat]
    """
    with out_path.open("w", encoding="utf-8") as f:
        for artist in sorted(repeats.keys()):
            f.write(f"- [{artist}, {repeats[artist]}]\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-artist repeat counts to balance training samples."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Root dataset directory containing subfolders per artist.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=1500,
        help="Artists with count > cap get repeat=1. Default: 1500.",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=10,
        help="Maximum repeat per artist (inclusive upper bound in constraint). Default: 10.",
    )
    parser.add_argument(
        "--target-effective",
        type=int,
        default=None,
        help="Manually set target effective samples per artist. If omitted, computed automatically.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path. If ends with .csv writes CSV; if .txt writes '- [artist, repeat]'. "
            "Default when omitted: project root 'repeats.txt'"
        ),
    )
    parser.add_argument(
        "--allow-1-and-10",
        action="store_true",
        help="Allow exact 1 and 10 repeats when rounding hits the bounds.",
    )
    args = parser.parse_args()

    artist_to_count = scan_dataset(args.dataset)
    repeats, target_effective = compute_repeats(
        artist_to_count=artist_to_count,
        cap=args.cap,
        max_repeat=args.max_repeat,
        target_effective=args.target_effective,
        avoid_exact_extremes=not args.allow_1_and_10,
    )

    mu, cv = summarize_effective(artist_to_count, repeats)
    out_path = args.out
    if out_path is None:
        out_path = Path(__file__).resolve().parent.parent / "repeats.txt"
    if out_path.suffix.lower() == ".txt":
        save_txt(out_path, repeats)
    else:
        save_csv(out_path, artist_to_count, repeats)

    print(f"Artists: {len(artist_to_count)}")
    print(f"Rule: counts > {args.cap} -> repeat=1; repeats in [1, {args.max_repeat}]")
    print(f"Target effective (used): {target_effective}")
    print(f"Mean effective count: {mu:.2f}, Coef. of variation (lower is better): {cv:.4f}")
    print(f"Saved: {out_path.resolve()}")
    print("\nTop 10 preview:")
    to_show = sorted(artist_to_count.items(), key=lambda kv: kv[1])[:10]
    for artist, c in to_show:
        print(f"- {artist}: count={c}, repeat={repeats[artist]}, effective={c*repeats[artist]}")


if __name__ == "__main__":
    main()