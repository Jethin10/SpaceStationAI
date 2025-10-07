import os, glob

root = "datasets/falcon_space"
def stems(split):
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(root, "images", split, "*.*"))}

train, val, test = stems("train"), stems("val"), stems("test")
print("Counts -> train:", len(train), "val:", len(val), "test:", len(test))

overlap = (train | val) & test
if overlap:
    print("ERROR: Some test images appear in train/val (first 10 shown):", list(sorted(overlap))[:10])
    raise SystemExit(1)
else:
    print("Good: test set is separate from train/val.")