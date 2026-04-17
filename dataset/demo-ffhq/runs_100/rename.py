#!/usr/bin/env python3
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite 000xx.png if it already exists")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without renaming")
    args = ap.parse_args()

    cwd = Path(".")
    prefix = "00000_run00"
    suffix = ".png"

    # Matches exactly: 00000_run00 + two chars + .png
    for src in sorted(cwd.glob("00000_run00??.png")):
        name = src.name
        xx = name[len(prefix): -len(suffix)]  # the ?? part

        # Hard-check: exactly two digits
        if len(xx) != 2 or not xx.isdigit():
            continue

        dst = cwd / f"000{xx}.png"

        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"{dst.name} already exists (from {src.name}). Use --overwrite if intended.")

        if args.dry_run:
            print(f"DRY: {src.name} -> {dst.name}")
        else:
            src.rename(dst)
            print(f"{src.name} -> {dst.name}")

if __name__ == "__main__":
    main()
