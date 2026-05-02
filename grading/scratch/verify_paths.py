"""Verify tất cả path trong src files resolve đúng."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # advanced_ai_project/

files_to_check = {
    "grading/src/detection/detector.py":    ("parents[2]", 2, "grading"),
    "grading/src/utils/loader.py":          ("parents[2]", 2, "grading"),
    "grading/src/utils/download.py":        ("parents[2]", 2, "grading"),
    "grading/src/utils/download_dataset.py":("parents[2]", 2, "grading"),
}

print("Verifying parents[N] for each src file:")
print("-" * 55)
all_ok = True
for rel_path, (label, n, expected_name) in files_to_check.items():
    f = ROOT / rel_path
    resolved = f.resolve().parents[n]
    ok = resolved.name == expected_name
    status = "[OK]" if ok else "[!!]"
    if not ok:
        all_ok = False
    print(f"  {status} {rel_path}")
    print(f"       {label} -> {resolved}  (expected: .../{expected_name}/)")

print()
print("Verifying actual model path exists:")
model_path = ROOT / "grading" / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"
if model_path.exists():
    mb = model_path.stat().st_size / 1024 / 1024
    print(f"  [OK] best.pt found ({mb:.1f} MB): {model_path}")
else:
    print(f"  [!!] best.pt NOT found: {model_path}")
    all_ok = False

print()
print("Verifying data/raw exists:")
raw_dir = ROOT / "grading" / "data" / "raw"
if raw_dir.exists():
    subdirs = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    print(f"  [OK] raw/ exists — {len(subdirs)} subdir(s)")
else:
    print(f"  [!!] raw/ NOT found: {raw_dir}")

print()
print("All checks passed!" if all_ok else "Some issues found above.")
