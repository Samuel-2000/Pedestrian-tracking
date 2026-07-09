## Resources & Downloads

Pre-trained models, documentation, example outputs and poster:

**[📁 Resources](https://drive.google.com/drive/u/1/folders/1zfFXMRVMnHY0xS7eI3EbHi3oTY3fBAH_)**

**Available Files:**
- `best_dataset1.pt`, `best_dataset2.pt`, `best_combined.pt` - Trained models
- `video.mp4` - Test input video
- `comparison_grid.mp4` - Example output video
- `Poster.pdf`, `PoVA_Project_Report.pdf` - Documentation
- `output/` folder - Generated graphs and tracking outputs

Datasets are automatically downloaded when you run `run.py`.

## Output Preview

![Output Preview](comparison_grid.gif)


## Installation

```bash
pip install -r requirements.txt
python run.py
```

The `run.py` script will:
- Install dependencies
- Download or train the models
- Setup datasets for testing
- Run tests and generate comparison graphs
- Create comparison video (if `input/video.mp4` exists)

## Quick Start

### Basic Tracking

```bash
python trackers/pedestrian_tracker.py \
    --input input/video.mp4 \
    --model models/dataset1/weights/best.pt \
    --tracker bytetrack
```

### With DeepSORT Tracker

```bash
python trackers/pedestrian_tracker.py \
    --input input/video.mp4 \
    --model models/combined/weights/best.pt \
    --tracker deepsort \
    --conf-thresh 0.3
```

**Parameters:**
- `--input` - Input video path (required)
- `--model` - Model path (e.g., `models/dataset1/weights/best.pt`)
- `--tracker` - `bytetrack` or `deepsort` (default: `bytetrack`)
- `--conf-thresh` - Confidence threshold 0.0-1.0 (default: `0.25`)
- `--output` - Output video path (default: `output.mp4`)

## Evaluation

### Test All Models

```bash
python testing.py
```

**Outputs:**
- `output/graphs/model_comparison.png` - Performance comparison charts
- `output/graphs/model_comparison_summary.txt` - Text summary

### MOT Dataset Evaluation

```bash
python tracker_evaluation.py
```
