"""Run a locally trained demo checkpoint on fictional inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from train_tp_radar_multitask import MultiTaskUNet, prepare_modalities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/prediction.npz'))
    args = parser.parse_args()
    torch.manual_seed(2026)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MultiTaskUNet(max_residual=state['max_residual'])
    model.load_state_dict(state['model'])
    model.eval()
    features = torch.rand(1, 12, 32, 32)
    background = features[:, 6:9].clone()
    inputs, background = prepare_modalities(features, background, False, 0, 0, 0)
    with torch.no_grad():
        rain_log, logits = model(inputs, background)
        rain = torch.expm1(rain_log).clamp_min(0).numpy()
        probability = logits.softmax(dim=2).numpy()
    if not np.isfinite(rain).all() or not np.isfinite(probability).all():
        raise ValueError('Nonfinite prediction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rain_mm=rain, class_probability=probability)
    print(f'Synthetic inference: rain={rain.shape}, probability={probability.shape}')


if __name__ == '__main__':
    main()
