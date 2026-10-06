"""Train one model on either hard (majority-vote) or soft (vote-distribution) labels.

Both label types use the same images, the same augmentation, the same
schedule and the same loss function; only the target changes. A hard label
is just a one-hot distribution, so one loss (cross-entropy against a target
distribution) covers both cases.

Run, for example:
    python -m src.train --model cnn --labels hard
    python -m src.train --model resnet18 --labels soft
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import NUM_CLASSES, RUNS_DIR
from src.data import get_split, load_processed
from src.models import build_model

DEFAULT_LR = {"cnn": 1e-3, "resnet18": 3e-4}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_cross_entropy(logits, target_probs):
    """Cross-entropy against a target distribution. With a one-hot target
    this is exactly the usual cross-entropy loss."""
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def augment(x, generator):
    """Random flip, small rotation, zoom and shift, applied on the GPU.
    x: (B, 1, 48, 48) in [0, 1]."""
    b = x.shape[0]
    rand = lambda *shape: torch.rand(*shape, device=x.device, generator=generator)
    angle = (rand(b) * 2 - 1) * math.radians(10)       # +-10 degrees
    scale = 1 + (rand(b) * 2 - 1) * 0.1                # 90%-110% zoom
    shift = (rand(b, 2) * 2 - 1) * 0.1                 # +-10% shift
    flip = torch.where(rand(b) < 0.5, -1.0, 1.0)       # horizontal flip
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, shift[:, 0]], dim=1),
        torch.stack([sin * flip, cos, shift[:, 1]], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def to_tensors(split, device):
    return {
        "x": torch.from_numpy(split["images"]).to(device).unsqueeze(1),   # uint8; scaled per batch
        "hard": torch.from_numpy(split["hard"]).to(device),
        "soft": torch.from_numpy(split["soft"]).to(device),
    }


@torch.no_grad()
def evaluate(model, data, batch_size=512):
    """Accuracy vs the majority label, and how far predictions are from the
    human vote distribution (cross-entropy and KL divergence)."""
    model.eval()
    correct, ce, kl, n = 0, 0.0, 0.0, len(data["hard"])
    for i in range(0, n, batch_size):
        x = data["x"][i:i + batch_size].float() / 255
        logits = model(x)
        logp = F.log_softmax(logits, dim=1)
        soft = data["soft"][i:i + batch_size]
        correct += (logits.argmax(1) == data["hard"][i:i + batch_size]).sum().item()
        ce += -(soft * logp).sum().item()
        kl += (soft * (torch.log(soft.clamp_min(1e-12)) - logp)).sum().item()
    return {"acc": correct / n, "soft_ce": ce / n, "kl": kl / n}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    p.add_argument("--labels", choices=["hard", "soft"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 for cnn, 3e-4 for resnet18")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=None, help="stop each epoch early (for quick tests)")
    p.add_argument("--out", default=None, help="output folder (default: runs/<model>_<labels>)")
    args = p.parse_args()
    lr = args.lr or DEFAULT_LR[args.model]

    set_seed(args.seed)
    device = get_device()
    out = Path(args.out) if args.out else RUNS_DIR / f"{args.model}_{args.labels}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | model: {args.model} | labels: {args.labels} | lr: {lr} | out: {out}")

    data = load_processed()
    train, val = to_tensors(get_split(data, "train"), device), to_tensors(get_split(data, "val"), device)
    if args.labels == "hard":
        targets = F.one_hot(train["hard"], NUM_CLASSES).float()
    else:
        targets = train["soft"]

    model = build_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    n_train = len(targets)
    batches_per_epoch = math.ceil(n_train / args.batch_size)
    if args.max_batches:
        batches_per_epoch = min(batches_per_epoch, args.max_batches)
    total_steps = args.epochs * batches_per_epoch
    warmup = batches_per_epoch   # 1 epoch of linear warmup, then cosine decay to 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    gen = torch.Generator(device=device).manual_seed(args.seed)
    order_gen = torch.Generator().manual_seed(args.seed)   # same image order for hard and soft runs

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        perm = torch.randperm(n_train, generator=order_gen).to(device)
        for b in range(batches_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            x = augment(train["x"][idx].float() / 255, gen)
            loss = soft_cross_entropy(model(x), targets[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        metrics = evaluate(model, val)
        metrics.update(epoch=epoch, train_loss=running / batches_per_epoch,
                       lr=sched.get_last_lr()[0], seconds=round(time.time() - start, 1))
        history.append(metrics)
        print(f"epoch {epoch:3d} | train loss {metrics['train_loss']:.4f} | val acc {metrics['acc']:.4f} "
              f"| val KL {metrics['kl']:.4f} | {metrics['seconds']}s")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    # We keep the final model (no early stopping on val), so neither label
    # type gets an advantage from a selection rule that favors it.
    torch.save({"model": args.model, "labels": args.labels, "state_dict": model.state_dict(),
                "args": vars(args) | {"lr": lr}}, out / "final.pt")
    print(f"Saved {out / 'final.pt'}")


if __name__ == "__main__":
    main()
