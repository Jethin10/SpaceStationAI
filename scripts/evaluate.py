import pandas as pd
import sys, pathlib

run_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/train/y8n_640_baseline")
res = pd.read_csv(run_dir / "results.csv")
print("Summary of last epoch:")
print(res.tail(1).T)

for p in ["confusion_matrix.png","PR_curve.png","F1_curve.png","P_curve.png","R_curve.png"]:
    fp = run_dir / "val" / p
    print(("FOUND " if fp.exists() else "MISSING"), p, "->", fp)