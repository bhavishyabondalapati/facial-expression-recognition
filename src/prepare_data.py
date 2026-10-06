"""Verify the raw CSVs and save a processed dataset to data/processed/ferplus.npz.

Run:  python -m src.prepare_data
"""
import numpy as np
import pandas as pd

from src.config import FER2013_CSV, FERPLUS_CSV, PROCESSED_DIR, PROCESSED_NPZ, TIE_BREAK_SEED
from src.data import build_dataset, verify_alignment


def main():
    print("Reading CSVs...")
    fer = pd.read_csv(FER2013_CSV)
    ferplus = pd.read_csv(FERPLUS_CSV)

    check = verify_alignment(fer, ferplus)   # raises if the files don't match
    print(f"Rows: {check['rows']} | Usage matches row by row")
    print(f"Original label agrees with FER+ top vote: {check['label_agreement']:.1%} "
          f"(if rows were shifted by one: {check['label_agreement_if_shifted']:.1%})")

    data = build_dataset(fer, ferplus, seed=TIE_BREAK_SEED)
    dropped = len(ferplus) - len(data["row"])
    print(f"Kept {len(data['row'])} images, dropped {dropped} (no image / not-a-face)")
    print(f"Hard-label ties broken at random (seed={TIE_BREAK_SEED}): {data['is_tie'].sum()}")
    for s in ["train", "val", "test"]:
        print(f"  {s:5s}: {(data['split'] == s).sum()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_NPZ, **data)
    print(f"Saved {PROCESSED_NPZ}")


if __name__ == "__main__":
    main()
