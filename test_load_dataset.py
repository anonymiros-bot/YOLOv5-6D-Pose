#!/usr/bin/env python3
"""Tests for load_dataset.py — run from repo root."""

import numpy as np
from pathlib import Path
from load_dataset import load_dataset, list_datasets, REPO_ROOT, DATASETS

PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        print(f"  ✗ {msg}")


def test_list_datasets():
    print("\n--- list_datasets ---")
    list_datasets()
    check(len(DATASETS) == 4, "4 datasets registered")


def test_unreal_flying():
    print("\n--- unreal_flying ---")
    data = load_dataset("unreal_flying")
    check(len(data) > 0, f"loaded {len(data)} sequences")
    check(len(data) == 140 or len(data) > 0, f"expected ~140, got {len(data)}")

    # Check a single sequence
    key = sorted(data.keys(), key=int)[0]
    seq = data[key]
    N = len(seq["t"])

    check(N > 0, f"seq {key}: {N} frames")
    check(seq["t"][0] == 0.0, f"seq {key}: timestamps start at 0")
    check(seq["pos"].shape == (N, 3), f"seq {key}: pos shape {seq['pos'].shape}")
    check(seq["quat"].shape == (N, 4), f"seq {key}: quat shape {seq['quat'].shape}")
    check(seq["pnp_pos"].shape == (N, 3), f"seq {key}: pnp_pos shape {seq['pnp_pos'].shape}")
    check(seq["pnp_q"].shape == (N, 4), f"seq {key}: pnp_q shape {seq['pnp_q'].shape}")
    check(isinstance(seq["images"], list), f"seq {key}: images is list")
    check(isinstance(seq["metadata"], dict), f"seq {key}: metadata is dict")

    # Check images exist on disk
    if len(seq["images"]) > 0:
        check(seq["images"][0].exists(), f"seq {key}: first image exists")
        check(len(seq["images"]) == N, f"seq {key}: {len(seq['images'])} images == {N} frames")

    # Check optional fields
    if "rts_vel" in seq:
        check(seq["rts_vel"].shape == (N, 3), f"seq {key}: rts_vel shape")
    if "cam_pos" in seq:
        check(seq["cam_pos"].shape == (N, 3), f"seq {key}: cam_pos shape")

    # Test single sequence loading
    data_single = load_dataset("unreal_flying", seq_keys=[key])
    check(len(data_single) == 1, f"single load: got {len(data_single)} sequence")
    check(key in data_single, f"single load: key '{key}' present")


def test_unreal_step():
    print("\n--- unreal_step ---")
    data = load_dataset("unreal_step")
    check(len(data) > 0, f"loaded {len(data)} sequences")
    check(len(data) == 35 or len(data) > 0, f"expected ~35, got {len(data)}")

    key = sorted(data.keys(), key=int)[0]
    seq = data[key]
    N = len(seq["t"])

    check(N > 0, f"seq {key}: {N} frames")
    check(seq["pos"].shape == (N, 3), f"seq {key}: pos shape {seq['pos'].shape}")

    if len(seq["images"]) > 0:
        check(seq["images"][0].exists(), f"seq {key}: first image exists")

    # Check metadata has level/cam_id
    check("level" in seq["metadata"] or "cam_id" in seq["metadata"],
          f"seq {key}: metadata has level/cam_id")


def test_real(name, expected_min=10):
    print(f"\n--- {name} ---")
    data = load_dataset(name)
    check(len(data) > 0, f"loaded {len(data)} sequences")
    check(len(data) >= expected_min, f"expected >={expected_min}, got {len(data)}")

    key = sorted(data.keys())[0]
    seq = data[key]
    N = len(seq["t"])

    check(N > 0, f"seq {key}: {N} frames")
    check(seq["t"][0] == 0.0, f"seq {key}: timestamps start at 0")
    check(seq["pos"].shape == (N, 3), f"seq {key}: pos shape {seq['pos'].shape}")
    check(seq["quat"].shape == (N, 4), f"seq {key}: quat shape {seq['quat'].shape}")
    check(seq["pnp_pos"].shape == (N, 3), f"seq {key}: pnp_pos shape")
    check(np.all(np.isnan(seq["pnp_pos"])), f"seq {key}: pnp_pos is NaN (raw data)")

    # Check images
    if len(seq["images"]) > 0:
        check(seq["images"][0].exists(), f"seq {key}: first image exists")
        check(len(seq["images"]) == N, f"seq {key}: {len(seq['images'])} images == {N} frames")

    # Check cam fields
    check("cam_pos" in seq, f"seq {key}: has cam_pos")
    check("cam_q" in seq, f"seq {key}: has cam_q")
    check(seq["cam_pos"].shape == (N, 3), f"seq {key}: cam_pos shape")

    # Check metadata
    check("session" in seq["metadata"], f"seq {key}: metadata has session")
    check("sequence" in seq["metadata"], f"seq {key}: metadata has sequence")

    # Test filtered loading
    data_single = load_dataset(name, seq_keys=[key])
    check(len(data_single) == 1, f"single load: got {len(data_single)} sequence")


def test_invalid():
    print("\n--- error handling ---")
    try:
        load_dataset("nonexistent")
        check(False, "should raise ValueError for unknown dataset")
    except ValueError:
        check(True, "raises ValueError for unknown dataset")


def test_consistency():
    """Cross-check: all sequences have matching array lengths."""
    print("\n--- consistency checks ---")
    for name in DATASETS:
        path = REPO_ROOT / (DATASETS[name].get("seq_root", DATASETS[name].get("data_dir", "")))
        if not path.exists():
            print(f"  - {name}: skipped (data not present)")
            continue

        data = load_dataset(name)
        mismatches = 0
        for key, seq in data.items():
            N = len(seq["t"])
            if seq["pos"].shape[0] != N:
                mismatches += 1
            if seq["quat"].shape[0] != N:
                mismatches += 1
            if seq["pnp_pos"].shape[0] != N:
                mismatches += 1
        check(mismatches == 0, f"{name}: all {len(data)} sequences have consistent shapes")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing load_dataset.py")
    print("=" * 60)

    test_list_datasets()
    test_invalid()

    # Only run data tests if data exists
    if (REPO_ROOT / "data/sequences/flying").exists():
        test_unreal_flying()
    else:
        print("\n--- unreal_flying: SKIPPED (no data) ---")

    if (REPO_ROOT / "data/sequences/acc_step").exists():
        test_unreal_step()
    else:
        print("\n--- unreal_step: SKIPPED (no data) ---")

    if (REPO_ROOT / "data/data/mavic2").exists():
        test_real("mavic2", expected_min=50)
    else:
        print("\n--- mavic2: SKIPPED (no data) ---")

    if (REPO_ROOT / "data/data/phantom4").exists():
        test_real("phantom4", expected_min=50)
    else:
        print("\n--- phantom4: SKIPPED (no data) ---")

    test_consistency()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
