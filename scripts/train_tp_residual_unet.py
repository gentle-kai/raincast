#!/usr/bin/env python
"""Train a regional-TP anchored residual U-Net on cached aligned samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class FusionCacheDataset(Dataset):
    def __init__(
        self,
        surface_root: Path,
        tp_root: Path,
        split: str,
        radar_root: Path | None = None,
        patch_size: int = 0,
        event_crop_prob: float = 0.0,
        random_crop: bool = True,
        limit: int | None = None,
    ) -> None:
        surface_dir = surface_root / split
        self.pairs = [(path, tp_root / split / path.name, radar_root / split / path.name if radar_root else None)
                      for path in sorted(surface_dir.rglob("*.npz"))]
        self.pairs = [(surface, tp, radar) for surface, tp, radar in self.pairs if tp.exists() and (radar is None or radar.exists())]
        if limit is not None:
            self.pairs = self.pairs[:limit]
        if not self.pairs:
            raise RuntimeError(f"No aligned cache pairs for split={split!r}")
        self.patch_size = patch_size
        self.event_crop_prob = event_crop_prob
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        surface_path, tp_path, radar_path = self.pairs[index]
        with np.load(surface_path) as item:
            obs = np.log1p(item["x_mm"].astype(np.float32))
            target = np.log1p(item["y_mm"].astype(np.float32))
            valid = item["y_valid"].astype(np.float32)
            land = (item["target_mask"] > 0).astype(np.float32)[None, :, :]
        with np.load(tp_path) as item:
            tp_log = np.log1p(item["tp_mm"].astype(np.float32))
        if radar_path is not None:
            with np.load(radar_path) as item:
                radar = np.clip(item["radar_dbz"].astype(np.float32), 0.0, 80.0) / 80.0
            features = np.concatenate([obs, tp_log, radar], axis=0)
        else:
            features = np.concatenate([obs, tp_log], axis=0)
        weight = valid * land

        if self.patch_size:
            _, height, width = features.shape
            if self.patch_size > min(height, width):
                raise ValueError(f"patch_size={self.patch_size} exceeds cached grid")
            rng = np.random.default_rng()
            candidates = None
            if self.random_crop and rng.random() < self.event_crop_prob:
                maximum = target.max(axis=0)
                for threshold in (np.log1p(50.0), np.log1p(20.0), np.log1p(5.0), np.log1p(0.1)):
                    found = np.argwhere((maximum >= threshold) & land[0].astype(bool))
                    if len(found):
                        candidates = found
                        break
            if candidates is not None:
                cy, cx = candidates[rng.integers(len(candidates))]
                top = int(np.clip(cy - self.patch_size // 2, 0, height - self.patch_size))
                left = int(np.clip(cx - self.patch_size // 2, 0, width - self.patch_size))
            elif self.random_crop:
                top = int(rng.integers(0, height - self.patch_size + 1))
                left = int(rng.integers(0, width - self.patch_size + 1))
            else:
                top = (height - self.patch_size) // 2
                left = (width - self.patch_size) // 2
            region = np.s_[:, top : top + self.patch_size, left : left + self.patch_size]
            features, target, tp_log, weight = features[region], target[region], tp_log[region], weight[region]
        return (torch.from_numpy(features), torch.from_numpy(target), torch.from_numpy(tp_log), torch.from_numpy(weight))


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TPResidualUNet(nn.Module):
    """Output is an additive residual in log1p precipitation space."""
    def __init__(self, input_channels: int = 9, base: int = 24, max_residual: float = 0.35) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.e1 = ConvBlock(input_channels, base)
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
        self.head = nn.Conv2d(base, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: Tensor, tp_log: Tensor) -> Tensor:
        e1 = self.e1(features)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        mid = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(mid), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return tp_log + self.max_residual * torch.tanh(self.head(d1))


def intensity_weight(target_log: Tensor, valid: Tensor) -> Tensor:
    rain = torch.expm1(target_log)
    multiplier = torch.full_like(rain, 0.12)
    multiplier = torch.where(rain >= 0.1, torch.ones_like(multiplier), multiplier)
    multiplier = torch.where(rain >= 5.0, torch.full_like(multiplier, 5.0), multiplier)
    multiplier = torch.where(rain >= 20.0, torch.full_like(multiplier, 25.0), multiplier)
    multiplier = torch.where(rain >= 50.0, torch.full_like(multiplier, 128.0), multiplier)
    return valid * multiplier


def loss_fn(prediction: Tensor, target: Tensor, valid: Tensor, rare_loss_weight: float) -> Tensor:
    weight = intensity_weight(target, valid)
    pointwise = F.smooth_l1_loss(prediction, target, beta=0.25, reduction="none")
    ordinary = (pointwise * valid).sum() / valid.sum().clamp_min(1.0)
    rare = (pointwise * weight).sum() / weight.sum().clamp_min(1.0)
    return ordinary + rare_loss_weight * rare


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, rare_loss_weight: float) -> float:
    model.eval()
    numerator = denominator = 0.0
    for features, target, tp_log, valid in loader:
        features, target, tp_log, valid = (item.to(device, non_blocking=True) for item in (features, target, tp_log, valid))
        loss = loss_fn(model(features, tp_log), target, valid, rare_loss_weight)
        batch_weight = float(intensity_weight(target, valid).sum())
        denominator += batch_weight
        numerator += float(loss) * batch_weight
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-cache", type=Path, required=True)
    parser.add_argument("--tp-cache", type=Path, required=True)
    parser.add_argument("--radar-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--event-crop-prob", type=float, default=0.7)
    parser.add_argument("--max-residual", type=float, default=0.35)
    parser.add_argument("--rare-loss-weight", type=float, default=0.20)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    train_set = FusionCacheDataset(args.surface_cache, args.tp_cache, "train", args.radar_cache, args.patch_size, args.event_crop_prob, True, args.max_train)
    valid_set = FusionCacheDataset(args.surface_cache, args.tp_cache, "offline_valid", args.radar_cache, args.patch_size, 0.0, False, args.max_valid)
    loader_args = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, **loader_args)
    model = TPResidualUNet(input_channels=12 if args.radar_cache else 9, max_residual=args.max_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"device={device}; train_pairs={len(train_set)}; valid_pairs={len(valid_set)}; parameters={sum(p.numel() for p in model.parameters()):,}")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for features, target, tp_log, valid in train_loader:
            features, target, tp_log, valid = (item.to(device, non_blocking=True) for item in (features, target, tp_log, valid))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(features, tp_log), target, valid, args.rare_loss_weight)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += loss.detach().item()
        valid_loss = evaluate(model, valid_loader, device, args.rare_loss_weight)
        print(f"epoch={epoch:03d} train_huber={total / len(train_loader):.6f} valid_huber={valid_loss:.6f}")
        if valid_loss < best:
            best = valid_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_huber": valid_loss,
                        "input_channels": 12 if args.radar_cache else 9, "max_residual": args.max_residual}, args.output_dir / "best.pt")
    print(f"best_valid_huber={best:.6f}; checkpoint={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
