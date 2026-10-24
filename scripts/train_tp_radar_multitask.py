#!/usr/bin/env python
"""Five-class, soft-TS, missing-modality-aware TP+radar model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from train_tp_residual_unet import ConvBlock, FusionCacheDataset  # noqa: E402

CLASS_WEIGHTS = (0.15, 1.0, 5.0, 25.0, 100.0)


class MultiTaskUNet(nn.Module):
    def __init__(self, base: int = 24, max_residual: float = 0.35) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.e1 = ConvBlock(15, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.regression = nn.Conv2d(base, 3, 1)
        self.classes = nn.Conv2d(base, 15, 1)
        nn.init.zeros_(self.regression.weight)
        nn.init.zeros_(self.regression.bias)

    def forward(self, features: Tensor, tp_log: Tensor) -> tuple[Tensor, Tensor]:
        e1 = self.e1(features)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        mid = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(mid), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        rain_log = tp_log + self.max_residual * torch.tanh(self.regression(d1))
        logits = self.classes(d1).view(d1.shape[0], 3, 5, d1.shape[2], d1.shape[3])
        return rain_log, logits


def class_target(target_log: Tensor) -> Tensor:
    rain = torch.expm1(target_log)
    # Example intensity bins are [0.1,5), [5,20), [20,50), [50,+inf).
    return torch.bucketize(
        rain, torch.tensor((0.1, 5.0, 20.0, 50.0), device=rain.device), right=True
    )


def prepare_modalities(
    features: Tensor, tp_log: Tensor, training: bool, obs_drop: float, tp_drop: float, radar_drop: float
) -> tuple[Tensor, Tensor]:
    batch, _, height, width = features.shape
    avail = torch.ones((batch, 3), device=features.device, dtype=features.dtype)
    if training:
        avail[:, 0] = (torch.rand(batch, device=features.device) >= obs_drop).to(features.dtype)
        avail[:, 1] = (torch.rand(batch, device=features.device) >= tp_drop).to(features.dtype)
        avail[:, 2] = (torch.rand(batch, device=features.device) >= radar_drop).to(features.dtype)
    features = features.clone()
    features[:, 0:6] *= avail[:, 0, None, None, None]
    features[:, 6:9] *= avail[:, 1, None, None, None]
    features[:, 9:12] *= avail[:, 2, None, None, None]
    tp_log = tp_log * avail[:, 1, None, None, None]
    masks = avail[:, :, None, None].expand(batch, 3, height, width)
    return torch.cat([features, masks], dim=1), tp_log


def losses(pred_log: Tensor, logits: Tensor, target_log: Tensor, valid: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    target_class = class_target(target_log)
    pointwise = F.smooth_l1_loss(pred_log, target_log, beta=0.25, reduction="none")
    regression = (pointwise * valid).sum() / valid.sum().clamp_min(1.0)

    ce_input = logits.permute(0, 2, 1, 3, 4)
    weights = torch.tensor(CLASS_WEIGHTS, device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(ce_input, target_class, weight=weights, reduction="none")
    probability = torch.softmax(logits, dim=2)
    pt = torch.gather(probability, 2, target_class.unsqueeze(2)).squeeze(2)
    focal = (((1.0 - pt) ** 2) * ce * valid).sum() / valid.sum().clamp_min(1.0)

    soft_ts_losses = []
    for lead in range(3):
        mask = valid[:, lead]
        for category in range(1, 5):
            truth = (target_class[:, lead] == category).to(logits.dtype) * mask
            pred = probability[:, lead, category] * mask
            if truth.sum() > 0:
                hit = (pred * truth).sum()
                union = pred.sum() + truth.sum() - hit
                soft_ts_losses.append(1.0 - hit / union.clamp_min(1.0))
    soft_ts = torch.stack(soft_ts_losses).mean() if soft_ts_losses else regression.new_zeros(())
    total = 0.30 * regression + focal + 0.75 * soft_ts
    return total, {"reg": regression, "focal": focal, "soft_ts": soft_ts}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums = {name: 0.0 for name in ("total", "reg", "focal", "soft_ts")}
    for features, target, tp_log, valid in loader:
        features, target, tp_log, valid = (x.to(device, non_blocking=True) for x in (features, target, tp_log, valid))
        model_input, base = prepare_modalities(features, tp_log, False, 0, 0, 0)
        pred, logits = model(model_input, base)
        total, parts = losses(pred, logits, target, valid)
        sums["total"] += total.item()
        for name in parts:
            sums[name] += parts[name].item()
    return {name: value / len(loader) for name, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-cache", type=Path, required=True)
    parser.add_argument("--tp-cache", type=Path, required=True)
    parser.add_argument("--radar-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--event-crop-prob", type=float, default=0.6)
    parser.add_argument("--obs-drop", type=float, default=0.05)
    parser.add_argument("--tp-drop", type=float, default=0.10)
    parser.add_argument("--radar-drop", type=float, default=0.10)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.benchmark = True
    train_set = FusionCacheDataset(args.surface_cache, args.tp_cache, "train", args.radar_cache,
                                   args.patch_size, args.event_crop_prob, True, args.max_train)
    valid_set = FusionCacheDataset(args.surface_cache, args.tp_cache, "offline_valid", args.radar_cache,
                                   args.patch_size, 0.0, False, args.max_valid)
    common = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **common)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, **common)
    model = MultiTaskUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"device={device}; train={len(train_set)}; valid={len(valid_set)}; parameters={sum(p.numel() for p in model.parameters()):,}")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for features, target, tp_log, valid in train_loader:
            features, target, tp_log, valid = (x.to(device, non_blocking=True) for x in (features, target, tp_log, valid))
            model_input, base = prepare_modalities(features, tp_log, True, args.obs_drop, args.tp_drop, args.radar_drop)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred, logits = model(model_input, base)
                total, _ = losses(pred, logits, target, valid)
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); running += total.detach().item()
        metrics = evaluate(model, valid_loader, device)
        print(f"epoch={epoch:03d} train={running/len(train_loader):.5f} valid={metrics['total']:.5f} "
              f"reg={metrics['reg']:.5f} focal={metrics['focal']:.5f} soft_ts={metrics['soft_ts']:.5f}")
        if metrics["total"] < best:
            best = metrics["total"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                        "class_weights": CLASS_WEIGHTS, "max_residual": model.max_residual}, args.output_dir / "best.pt")
    print(f"best_valid={best:.6f}; checkpoint={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
