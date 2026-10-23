# Football Shot Intelligence

**Football Shot Intelligence** is a computer-vision and deep-learning proof of concept for automatically detecting football shot events and extracting short highlight clips from broadcast video.

The project combines a football-specific YOLO object detector with temporal ball-motion analysis and simple spatial event reasoning.

## What it does

The current pipeline:

1. detects the football and players in video frames,
2. tracks the ball position through time,
3. finds likely player–ball contact events,
4. filters physically implausible detections,
5. distinguishes a shot candidate from other kicks using spatial and directional context,
6. automatically extracts a short video clip around the detected shot.

The v1 proof of concept was tested on a short broadcast clip containing Zinedine Zidane's volley in the 2002 UEFA Champions League final.

The system automatically identified the shot at approximately **7.57 seconds** and extracted the corresponding highlight clip.

---

## Motivation

Generic object detectors are not optimized specifically for football broadcast footage.

The first baseline used a COCO-pretrained YOLO11s model. Player detection worked reasonably well, but football detection was unstable because the ball is a very small object in wide broadcast frames.

The project therefore follows this workflow:

```text
Generic YOLO baseline
        ↓
Identify small-ball detection problem
        ↓
Football-specific annotated dataset
        ↓
YOLO transfer learning / fine-tuning
        ↓
Football-specific detector
        ↓
Temporal ball analysis
        ↓
Kick detection
        ↓
Shot classification
        ↓
Automatic clip extraction
```

---

## Deep-learning workflow

### Baseline

The initial model was:

```text
YOLO11s
COCO pretrained weights
input size: 1280
```

The generic detector uses COCO classes such as:

```text
person
sports ball
```

Testing on football broadcast footage showed that player detection was usable, while ball detection frequently failed around the actual shot.

### Football-specific fine-tuning

YOLO11s was then fine-tuned using the open **Football Image Dataset**, which contains football-specific object annotations.

The custom detector uses three classes:

```text
0 — ball
1 — player
2 — referee
```

Training configuration:

```text
Base model: YOLO11s
Transfer learning: COCO pretrained weights
Epochs: 30
Image size: 1280
Batch size: 4
Hardware: NVIDIA Tesla T4
```

The resulting checkpoint is stored as:

```text
models/football_detector.pt
```

This represents transfer learning rather than training a detector from scratch.

---

## Event detection pipeline

### 1. Object detection

For every frame, the fine-tuned YOLO model predicts football-specific bounding boxes.

Relevant detections are reduced to:

```text
ball
player
```

### 2. Ball trajectory

For each detected ball, the system stores its image-space centre:

```text
(frame, time, ball_x, ball_y)
```

This produces a temporal ball trajectory.

### 3. Player–ball relationship

For each frame, the nearest player to the ball is estimated using the bottom-centre of the player's bounding box.

The pipeline therefore derives:

```text
ball position
ball speed in pixel space
distance to nearest player
```

### 4. Kick detection

A potential kick is identified when:

```text
player is close to the ball
        +
ball subsequently moves away
        +
departure velocity is sufficiently high
```

Large physically implausible jumps are rejected as likely detection errors.

In the demonstration sequence, the initial detector produced seven possible contacts.

After filtering implausible ball jumps, three meaningful football contacts remained:

```text
2.43 s — KICK
5.80 s — KICK
7.57 s — KICK
```

### 5. Shot classification

The current MVP adds simple spatial context:

```text
kick
+
advanced attacking position
+
ball displacement toward goal side
=
SHOT
```

For the demonstration clip the final result was:

```text
2.43 s — KICK
5.80 s — KICK
7.57 s — SHOT
```

The shot was then automatically extracted into a short video clip.

---

## Project structure

```text
football-shot-intelligence/
│
├── models/
│   └── football_detector.pt
│
├── scripts/
│   ├── analyze_ball.py
│   ├── detect_kicks.py
│   └── detect_shots.py
│
├── data/
│   ├── input/
│   └── output/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### `analyze_ball.py`

Runs the fine-tuned YOLO detector and produces:

```text
annotated video
ball trajectory data
nearest-player distances
pixel-space ball velocity
CSV analysis
```

### `detect_kicks.py`

Uses the temporal analysis to detect meaningful player–ball contact events and reject implausible detection jumps.

### `detect_shots.py`

Adds attacking-direction and spatial logic to classify shot candidates and automatically extracts the corresponding video clips.

---

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the pipeline

Place a local football video at:

```text
data/input/zidane_goal.mp4
```

The input footage is intentionally not included in this repository.

Run football detection and temporal analysis:

```powershell
python scripts/analyze_ball.py
```

Detect player–ball contact events:

```powershell
python scripts/detect_kicks.py
```

Detect shots and extract clips:

```powershell
python scripts/detect_shots.py
```

Generated files are written under:

```text
data/output/
```

---

## Example result

For the demonstration sequence:

```text
EVENT CANDIDATES
================

#1 KICK
Time: 2.43s

#2 KICK
Time: 5.80s

#3 SHOT
Time: 7.57s

Shots detected: 1
```

The pipeline automatically generated:

```text
data/output/shots/shot_001_7.57s.mp4
```

---

## Technical stack

- Python
- PyTorch
- Ultralytics YOLO11
- OpenCV
- Pandas
- Transfer learning
- Object detection
- Temporal computer vision
- Video processing

---

## Limitations

This repository is intentionally a compact proof of concept rather than a production football event-recognition system.

The current v1 has several important limitations:

- attacking direction is provided as a sequence-level assumption;
- the advanced attacking zone is estimated in image coordinates rather than calibrated pitch coordinates;
- ball velocity is measured in pixels per second, not physical units;
- the shot classifier currently uses interpretable geometric rules rather than a learned temporal classifier;
- camera cuts and major broadcast-angle changes are not explicitly handled;
- the demonstration has not yet been benchmarked across a large collection of complete matches.

The current goal is to demonstrate the full engineering pipeline from football-specific deep-learning detection to automatic event extraction.

---

## Possible next steps

Potential extensions include:

- automatic goal and attacking-direction detection,
- pitch calibration and homography,
- physical ball-speed estimation,
- player tracking,
- automatic team classification,
- jersey-number recognition,
- learned pass / cross / shot classification,
- processing complete matches,
- searchable shot timelines and shot maps.

---

## Dataset

The football-specific detector was fine-tuned using:

**Football Image Dataset**  
https://github.com/skaczylo/Football-Image-Dataset

The dataset provides YOLO-format annotations for:

```text
ball
player
referee
```

The dataset is distributed under the **CC BY 4.0** license. Please refer to the original repository for full attribution and licensing information.

---

## Video footage

Broadcast match footage used during development is **not distributed with this repository**.

Users must provide their own video input and ensure they have the appropriate rights to use it.

---

## Status

**v1 — proof of concept complete**

Current validated flow:

```text
football video
→ fine-tuned YOLO
→ ball/player detection
→ temporal analysis
→ kick detection
→ shot classification
→ automatic highlight extraction
```

## Author

Tomislav Ivanić

Software / Computer Vision Engineer