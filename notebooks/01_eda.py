"""Quick EDA over the built M1 dataset. Run after pipelines/build_dataset.py."""
from pathlib import Path
import pandas as pd

PROC = Path("data/processed")

def main():
    manifest = pd.read_parquet(PROC / "manifest.parquet")
    print("Cases:", len(manifest), "| kept:", int(manifest["kept"].sum()))
    print("Duration (samples) describe:\n", manifest["n_samples"].describe())

    case_files = sorted((PROC / "cases").glob("case_*.parquet"))
    frames = [pd.read_parquet(f) for f in case_files]
    if not frames:
        print("No case files — run the pipeline first."); return
    allrows = pd.concat(frames, ignore_index=True)
    for col in ["bis", "map", "propofol_rate", "remifentanil_rate", "hr", "spo2"]:
        if col in allrows.columns:
            s = allrows[col]
            print(f"{col:18s} mean={s.mean():7.2f} p5={s.quantile(.05):7.2f} "
                  f"p95={s.quantile(.95):7.2f} nan%={s.isna().mean()*100:5.2f}")

    in_target = allrows["bis"].between(40, 60).mean()
    print(f"Time in BIS target [40,60]: {in_target*100:.1f}%")

if __name__ == "__main__":
    main()
