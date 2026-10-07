"""Real-time facial expression demo.

Finds faces with OpenCV's Haar detector, crops each one to 48x48 grayscale
(like FER2013), and shows the model's probabilities for all 8 emotions.
Because the default model was trained on soft labels, the bars approximate
how a group of people would split their votes, not just a single answer.

Run:
    python -m src.webcam                          # webcam, soft-label CNN
    python -m src.webcam --model resnet18_soft    # use another trained run
    python -m src.webcam --image photo.jpg        # annotate a photo instead
Press q or Esc to quit.
"""
import argparse
import time

import cv2
import numpy as np
import torch

from src.config import CLASSES, IMG_SIZE, RUNS_DIR
from src.models import build_model
from src.train import get_device

# BGR colors for OpenCV drawing
BAR_COLOR = (52, 104, 235)      # orange: matches the "soft" color in the charts
TEXT = (255, 255, 255)
PANEL = (30, 30, 30)
BOX = (214, 120, 42)            # blue face box
UNCERTAIN_BELOW = 0.5           # top probability under this -> show the top two


def load_model(run, device):
    ckpt = torch.load(RUNS_DIR / run / "final.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def preprocess_face(gray, box, margin=0.0):
    """Crop a face box from a grayscale frame -> float tensor (1, 1, 48, 48) in [0, 1].
    margin grows the box by that fraction on each side."""
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    x0, y0 = max(x - dx, 0), max(y - dy, 0)
    x1, y1 = min(x + w + dx, gray.shape[1]), min(y + h + dy, gray.shape[0])
    face = cv2.resize(gray[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(face).float().div(255).view(1, 1, IMG_SIZE, IMG_SIZE)


class Smoother:
    """Exponential moving average of probabilities, so bars don't flicker
    from frame to frame. alpha = weight of the newest frame."""

    def __init__(self, alpha=0.3):
        self.alpha, self.value = alpha, None

    def update(self, probs):
        self.value = probs if self.value is None else self.alpha * probs + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


def describe(probs):
    """Short label for the face box, e.g. 'happiness 82%' or
    'uncertain: neutral 41% / sadness 33%' when the model is unsure."""
    order = np.argsort(probs)[::-1]
    top, second = order[0], order[1]
    if probs[top] < UNCERTAIN_BELOW:
        return f"uncertain: {CLASSES[top]} {probs[top]:.0%} / {CLASSES[second]} {probs[second]:.0%}"
    return f"{CLASSES[top]} {probs[top]:.0%}"


def draw_bars(frame, probs, x0=10, y0=10, width=220):
    """Panel with one bar per emotion."""
    row = 24
    cv2.rectangle(frame, (x0, y0), (x0 + width + 110, y0 + row * len(CLASSES) + 12), PANEL, -1)
    for i, (name, p) in enumerate(zip(CLASSES, probs)):
        y = y0 + 10 + i * row
        cv2.putText(frame, name, (x0 + 8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 95, y + 3), (x0 + 95 + int(p * width), y + 17), BAR_COLOR, -1)
        cv2.putText(frame, f"{p:.0%}", (x0 + 100 + width, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1,
                    cv2.LINE_AA)


def draw_face(frame, box, label):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), BOX, -1)
    cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 2, cv2.LINE_AA)


class Demo:
    def __init__(self, run, margin):
        self.device = get_device()
        self.model = load_model(run, self.device)
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.margin = margin
        self.smoother = Smoother()

    def detect(self, gray):
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return sorted((tuple(int(v) for v in f) for f in faces), key=lambda f: f[2] * f[3], reverse=True)

    @torch.no_grad()
    def predict(self, gray, box):
        x = preprocess_face(gray, box, self.margin).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    def annotate(self, frame, smooth=True):
        """Draw predictions on frame (in place). Bars show the largest face;
        smoothing applies to it only, since we don't track faces between frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect(gray)
        if not faces:
            self.smoother.reset()
            cv2.putText(frame, "no face found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)
            return frame
        for i, box in enumerate(faces):
            probs = self.predict(gray, box)
            if i == 0:
                probs = self.smoother.update(probs) if smooth else probs
                draw_bars(frame, probs)
            draw_face(frame, box, describe(probs))
        return frame


def run_webcam(demo, camera):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. On macOS, allow camera access for your terminal app in "
                         "System Settings > Privacy & Security > Camera, then restart the terminal.")
    last = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)   # mirror view, like a selfie camera
            demo.annotate(frame)
            now = time.time()
            fps, last = 1 / max(now - last, 1e-6), now
            cv2.putText(frame, f"{fps:.0f} fps | q to quit", (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
            cv2.imshow("Facial expression (FER+)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="cnn_soft", help="run folder in runs/ (default: cnn_soft)")
    p.add_argument("--image", help="annotate this image instead of using the webcam")
    p.add_argument("--out", help="where to save the annotated image (default: <image>_annotated.png)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--margin", type=float, default=0.0, help="extra space around detected faces (e.g. 0.1)")
    args = p.parse_args()

    demo = Demo(args.model, args.margin)
    print(f"Model: {args.model} on {demo.device}")
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        demo.annotate(frame, smooth=False)
        out = args.out or args.image.rsplit(".", 1)[0] + "_annotated.png"
        cv2.imwrite(out, frame)
        print(f"Saved {out}")
    else:
        run_webcam(demo, args.camera)


if __name__ == "__main__":
    main()
