# ASCII-Driven Hybrid Deepfake Detection Framework

## Structure
- `src/preprocessing`: face detection, frame sampling, ASCII conversion
- `src/models`: Inception-ResNet V2 + EfficientNet-B4 + LSTM + attention fusion
- `src/training`: composite loss, optimizer/scheduler, trainer
- `src/evaluation`: metrics, benchmark, visualizations
- `data`: dataset and transforms
- `scripts`: train/evaluate/inference entrypoints

## Quickstart
```bash
cd deepfake_detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

## Notes
- CPU-only execution supported via `device: cpu` in config.
- Populate dataset video paths/labels in `configs/default.yaml`.
