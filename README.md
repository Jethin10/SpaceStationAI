# SpaceStationAI — Duality AI Safety Object Detection

Quickstart
1) Create/activate env: Windows: run setup_env.bat, then `conda activate EDU`
2) Put dataset here (not in git):
   - datasets/train3/{images,labels}
   - datasets/val3/{images,labels}
3) Configs: `configs/data.yaml` points to `../datasets`
4) Smoke test:
   `python scripts/train.py --data configs/data.yaml --hyp configs/hyp_falcon.yaml --epochs 1 --batch 4 --imgsz 640 --workers 0`
5) Train:
   `python scripts/train.py --data configs/data.yaml --hyp configs/hyp_falcon.yaml --epochs 100 --batch 16 --imgsz 640 --workers 0`