"""
DP Segmentation with AdaDPS Preconditioning.

Investigates whether public auxiliary data can improve DP-SGD training
for medical image segmentation via gradient preconditioning.

Key idea (AdaDPS - ICML'22):
- Estimate diagonal preconditioner E[g²] from public data
- Apply g / sqrt(E[g²] + eps) BEFORE gradient clipping
- This normalizes gradient scales, making clipping more uniform

4 Comparison Methods:
  1. baseline        - No DP (upper bound on utility)
  2. dp_no_precond   - DP-SGD without preconditioning
  3. dp_adadps_public - DP-SGD + preconditioner from public data (DRIVE)
  4. dp_adadps_oracle - DP-SGD + preconditioner from private data (oracle)

The hypothesis: Method 3 should approach Method 4's performance while
maintaining privacy (public data doesn't leak private information).
"""

from pathlib import Path


def main():
    print("=" * 60)
    print("DP Segmentation with AdaDPS Preconditioning")
    print("=" * 60)
    print()
    print("4 Comparison Methods:")
    print("  1. baseline         - No DP (upper bound)")
    print("  2. dp_no_precond    - DP-SGD only")
    print("  3. dp_adadps_public - DP-SGD + AdaDPS from public data")
    print("  4. dp_adadps_oracle - DP-SGD + AdaDPS from private (oracle)")
    print()

    # Check data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    drive = data_dir / "DRIVE"
    stare = data_dir / "STARE"

    print("Datasets:")
    if drive.exists():
        n = len(list((drive / "training" / "images").glob("*.png")))
        print(f"  DRIVE (public): {n} images")
    else:
        print("  DRIVE: Not found - run preprocessing")

    if stare.exists():
        n = len(list((stare / "images").glob("*.png")))
        print(f"  STARE (private): {n} images")
    else:
        print("  STARE: Not found - run preprocessing")

    print()
    print("Usage:")
    print("  # Run single method")
    print("  uv run python scripts/train.py --method dp_adadps_public")
    print()
    print("  # Run all methods")
    print("  uv run python scripts/run_experiments.py")
    print()
    print("  # Test different epsilons")
    print("  uv run python scripts/run_experiments.py --epsilons 1.0 2.0 4.0 8.0")


if __name__ == "__main__":
    main()
