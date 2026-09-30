# Why Jersey Numbers Are Hard to Read in Hockey: An End-to-End Player Tracking and Identification Pipeline for PWHL Broadcast Video

An end-to-end player tracking and jersey number identification pipeline, following an
earlier single-frame detection project into full tracklet-based tracking — built and
evaluated on PWHL broadcast footage, and compared directly against NHL footage from the
McGill Hockey Player Tracking Dataset. Draws on Vats, Walters, Fani, Clausi, and Zelek's
*"Player Tracking and Identification in Ice Hockey"* and Koshkina's jersey-number-recognition
pipeline (legibility classifier + PARSeq), used as reference implementations tested against
PWHL data rather than prior work being directly followed up.

## Motivation

This project picks up where [the first one](https://github.com/rebeca-meraz/pwhl-cv-pipeline) left off. That project
covered single-frame detection and OCR; its own conclusion named the logical next steps —
a properly-sized labeled dataset, a second OCR engine for comparison, or a purpose-trained
model instead of a general-purpose one. This project pursues the last of those, while also
extending single-frame detection into full player tracking: detecting players across video,
tracking them as tracklets, filtering out referees, identifying teams, and reading jersey
numbers by aggregating evidence across each tracklet rather than a single frame. The
broader motivation is unchanged — tracking connects to fatigue, fatigue connects to
performance, and performance connects to coaching decisions — this project is about whether
the building blocks hold up well enough to build on.

## Main Contributions

- Built a complete tracklet-based pipeline — tracking, referee filtering, team
  identification, and jersey-number reading — evaluated end-to-end against a manually
  labeled test set cross-checked against official team rosters.
- Diagnosed a silent, high-impact pipeline bug: two downstream models expect different
  crops of the same player (full-body for the legibility classifier, pose-cropped torso
  for PARSeq), and conflating them dropped reading accuracy to 0% before being caught and
  fixed.
- Trained a referee classifier (ResNet18, transfer learning) after three
  hand-crafted heuristics (color saturation, grayscale variance, stripe-crossing counts)
  all failed to separate referees from players — validated with leave-one-game-out across
  5 different games/stadiums (91.3–98.0% accuracy on unseen games).
- Ran a direct comparison against NHL broadcast footage (the dataset Koshkina's model was
  actually trained on), providing evidence consistent with a possible domain gap between
  NHL and PWHL broadcast styles, rather than treating PWHL underperformance as unexplained.
- Connected back to the first project's findings quantitatively: the 31.6% tracklet-level
  accuracy obtained here, versus 9.7% single-frame exact-match accuracy previously, is
  contextual evidence that temporal/pipeline-level aggregation recovers useful identity
  information — though the two metrics use different evaluation protocols and aren't a
  direct improvement measurement.

## Methodology

| Stage | What happened |
|---|---|
| Video acquisition | Re-acquired the same broadcast source in 1080p (vs. 360p in the first project) after finding it substantially improved crop quality |
| Tracking | BoT-SORT (with camera motion compensation) + PySceneDetect scene-cut detection; BoT-SORT reduced tracklet fragmentation from 153 to 102 on the same clip compared to ByteTrack |
| Referee filtering | ResNet18 classifier, transfer learning, trained on ~1,700 hand-labeled examples across 5 games; leave-one-game-out validated |
| Team identification | K-means clustering (2 clusters) on average LAB color of the pose-cropped torso; single-frame color was unstable (63% agreement with confirmed labels), fixed by averaging over 5 frames per tracklet (89.5%) |
| Jersey number reading | Koshkina's legibility classifier (full-body crops) → pose-cropped torso (YOLOv8-pose) → hockey-fine-tuned PARSeq → confidence-weighted majority vote per tracklet |
| Evaluation | 19-tracklet PWHL test set (8 known players, cross-checked against team rosters) + 8-tracklet NHL test set for domain comparison |

## Repo Structure

```
├── 01_referee_classifier.ipynb
├── 02_PWHL_Tracking_ID.ipynb
└── README.md
```

## Results

| Component | Result |
|---|---|
| BoT-SORT vs ByteTrack | 102 vs 153 tracklets (fragmentation only) |
| Referee filtering | 91.3–98.0% |
| Team ID | 89.5% |
| PWHL jersey reading | 31.6% |
| NHL jersey reading | 50.0% |

## Limitations

- **Small test sets** (19 PWHL tracklets, 8 NHL tracklets) — accuracy figures are
  directional evidence, not precise estimates.
- **31.6% jersey-reading accuracy on PWHL**, well below Koshkina's published 91.4% — but
  that figure is image-level, evaluated only on already-legible images, not directly
  comparable to this project's tracklet-level accuracy without matching methodology.
- **The domain-gap hypothesis is suggestive, not confirmed** — camera angle, broadcast
  style, and resolution are confounded between the NHL and PWHL comparisons and weren't
  tested independently.
- **Team ID and the full pipeline were validated on a single clip**; the referee classifier
  on 5 games. Performance on the full 22-game PWHL dataset is untested.
- **Several individual failures reflect genuine source-image ambiguity**, not fixable
  pipeline bugs — a folded jersey obscuring a digit, digit shapes ambiguous even to a
  human at this resolution.
- PARSeq's confidence scores were not recalibrated for this domain, and were repeatedly
  unreliable as a standalone filter.

## Personal Reflection

The most useful pattern from this project wasn’t any single fix — it was understanding why jersey numbers are so difficult to read in hockey: the visual evidence needed for identification is often degraded by viewpoint, posture, motion blur, resolution, and occlusion, while pipeline choices can further determine whether the remaining information is actually used. Hand-crafted heuristics consistently struggled against this variability — color saturation, stripe detection, and single-frame color clustering all had to be replaced or strengthened with learned or aggregated approaches. Just as importantly, checking results visually rather than trusting a metric or confidence score was what actually exposed the failures worth fixing. The main lesson I want to carry into the next step of this chain of questions is that in computer vision, understanding what information the image actually contains — and where that information can be lost — can matter as much as choosing the model that processes it.

## Tools

Python, YOLOv8 + YOLOv8-Pose (Ultralytics), BoT-SORT (`trackers`), PySceneDetect, PyTorch/
torchvision (ResNet18 transfer learning), PARSeq, scikit-learn (K-means), OpenCV, pandas,
Matplotlib, ipywidgets — built in Jupyter, with video acquired via `yt-dlp` and `ffmpeg`.
