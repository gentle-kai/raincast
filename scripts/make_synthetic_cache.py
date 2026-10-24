"""Generate random aligned caches for a software demonstration."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    for split, count in [('train', 4), ('offline_valid', 2)]:
        for kind in ('surface', 'tp', 'radar'):
            (args.output_dir / kind / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            name = f'synthetic_{split}_{i:03d}.npz'
            target = rng.choice(np.array([0, 1, 10, 30, 60], np.float32),
                                size=(3, 32, 32), p=[.6, .2, .12, .06, .02])
            obs = rng.gamma(1, 2, (6, 32, 32)).astype(np.float32)
            tp = np.maximum(target + rng.normal(0, 3, target.shape), 0).astype(np.float32)
            radar = rng.uniform(0, 60, (3, 32, 32)).astype(np.float32)
            np.savez_compressed(args.output_dir / 'surface' / split / name,
                                x_mm=obs, y_mm=target, y_valid=np.ones_like(target),
                                target_mask=np.ones((32, 32), np.uint8))
            np.savez_compressed(args.output_dir / 'tp' / split / name, tp_mm=tp)
            np.savez_compressed(args.output_dir / 'radar' / split / name, radar_dbz=radar)
    print('Synthetic caches created. No scientific evaluation is implied.')


if __name__ == '__main__':
    main()
