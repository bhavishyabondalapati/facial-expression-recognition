# Facial Expression Recognition: Hard vs Soft Labels on FER+

Each face in FER+ was labeled by 10 people, and they often disagree. This project
compares a model trained on the **majority vote** (one "correct" emotion per face)
against one trained on the **full vote distribution** (e.g. 60% happy, 40% surprise),
to see which one handles ambiguous faces better. It uses a small CNN trained from
scratch and an ImageNet-pretrained ResNet18, keeps all 8 emotion classes, trains on
Apple Silicon (`mps`), and will end with a real-time webcam demo.

> **Status:** data pipeline, exploration and training code are done.
> Training runs, evaluation and the webcam demo are next.

## Tools and libraries

| Tool | Why |
|---|---|
| **PyTorch + torchvision** | Training on the Mac GPU via `mps`; torchvision provides the pretrained ResNet18 |
| **pandas / NumPy** | Reading the CSVs and building label arrays |
| **matplotlib** | Exploration and result charts |
| **scikit-learn** | Evaluation metrics (confusion matrix, per-class F1) |
| **OpenCV** | Face detection and camera capture for the webcam demo |
| **pytest** | Tests for the label logic, models and loss |

## File structure

```
facial-expression-recognition/
├── data/                         # not committed (dataset license; too large)
│   ├── raw/fer2013.csv           # 35,887 faces as pixel strings + official split
│   ├── raw/fer2013new.csv        # FER+ crowd votes (10 annotators per face)
│   └── processed/ferplus.npz     # images + hard/soft labels, built by prepare_data
├── results/
│   ├── data_summary.md           # class counts, ambiguity stats, contempt, duplicates
│   └── figures/                  # exploration charts (see below)
├── runs/                         # not committed: checkpoints + training history
├── src/
│   ├── config.py                 # paths, class names, constants
│   ├── data.py                   # alignment check, hard/soft label building, loading
│   ├── prepare_data.py           # script: verify CSVs, save ferplus.npz
│   ├── explore.py                # script: exploration figures + data_summary.md
│   ├── models.py                 # scratch CNN and ResNet18 (both take 48x48 grayscale)
│   └── train.py                  # script: train one model on hard or soft labels
├── tests/
│   ├── test_data.py              # label building, tie-breaks, row dropping, alignment
│   └── test_models.py            # output shapes, loss correctness, augmentation
├── requirements.txt
└── README.md
```

## Setup

1. Create the environment:
   ```bash
   conda create -n facial-expression-recognition python=3.11 -y
   conda activate facial-expression-recognition
   pip install -r requirements.txt
   ```
2. Get the data and put both files in `data/raw/`:
   - `fer2013.csv` from the Kaggle FER2013 challenge (or a mirror of that same CSV;
     **not** the version with images sorted into folders, which loses row order)
   - `fer2013new.csv` from https://github.com/microsoft/FERPlus

## How to run

```bash
python -m src.prepare_data      # verify the CSVs line up, build data/processed/ferplus.npz
python -m src.explore           # figures + results/data_summary.md
python -m src.train --model cnn --labels hard      # one of 4 training runs
python -m pytest                # run the tests
```

The four runs are every combination of `--model {cnn,resnet18}` and `--labels {hard,soft}`.

## How it was built

1. **Checked the data lines up.** FER+ votes are matched to FER2013 images by row
   position only, so `prepare_data` checks the row counts and the `Usage` column row
   by row. As an extra check, the original FER2013 label agrees with the FER+ top
   vote on 62.8% of rows; shifting by one row drops that to 18% (chance).
2. **Built both label types on the same images.** We dropped 178 rows with no image
   or a top vote of "not a face". For every remaining image:
   - *soft label* = the 8 emotion vote counts divided by their total
   - *hard label* = the emotion with the most votes; the 1,668 ties (4.7%) are broken
     by a random choice with a fixed seed, so both models see exactly the same images
3. **Explored the data** (figures below).
4. **Wrote the training code.** Hard and soft runs share images, image order,
   augmentation, learning-rate schedule and loss. A hard label is a one-hot
   distribution, so one loss function handles both.
5. *Next:* train the 4 models, evaluate on clear vs ambiguous faces, build the webcam demo.

### What the data looks like

| | |
|---|---|
| ![class counts](results/figures/class_counts.png) | ![agreement](results/figures/ambiguity_hist.png) |
| ![agreement by class](results/figures/agreement_by_class.png) | ![hard vs soft](results/figures/hard_vs_soft_label_mass.png) |

- Only **26%** of faces get a unanimous vote; on **15%**, the top emotion gets 5 or fewer of 10 votes.
- **Contempt** is the majority label for just 266 of 35,709 images (0.74%): 200 in train
  and 34 in test. Annotators also agree with each other less on contempt (56% on average).
  Expect low per-class scores for contempt from any model. With 34 test images,
  its scores will also be noisy.
- Soft labels give minority emotions more total training weight. Contempt's share
  goes from 0.7% (hard) to 1.8% (soft), because 4,006 images got at least one contempt vote.

### Data quirks worth knowing

- **Duplicate images across splits:** 279 val and 288 test images have a pixel-identical
  copy in train. We keep the official splits so the numbers compare with published work.
- **Reused filenames in `fer2013new.csv`:** 4 image names appear twice, and 2 of those
  pairs are different faces. This doesn't affect us, because we match by row, not by name.

## Key concepts

- **Hard vs soft labels:** a hard label says "this face is happy". A soft label says
  "6 of 10 people said happy, 4 said surprise". Soft labels keep the information about
  how ambiguous a face is.
- **Cross-entropy with a target distribution:** loss = −Σ target × log(prediction).
  With a one-hot target, this reduces to normal cross-entropy. That's why the two
  experiments can share one loss function.
- **KL divergence:** how far the model's predicted distribution is from the human vote
  distribution (0 = identical). It measures how well the model captures ambiguity,
  which accuracy can't.
- **Controlled experiment:** to credit the labels with any difference, everything else
  (images, order, augmentation, schedule, seed, number of epochs) is held fixed.
  We keep the final epoch's model instead of picking the "best" epoch, because the
  rule for "best" could favor one label type.
- **Transfer learning:** ResNet18 starts with features learned on ImageNet photos. We
  upscale the 48×48 grayscale face and copy it into 3 channels to match what it expects.
- **Class imbalance:** contempt and disgust have about 200 training examples each, versus
  about 10,000 for neutral. Overall accuracy hides how badly rare classes do, so we
  report per-class F1 too.
- **Data leakage:** duplicate images across train and test can make test scores look
  better than real-world performance.
