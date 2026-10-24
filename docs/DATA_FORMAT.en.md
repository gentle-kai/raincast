# Aligned Cache Interface

[简体中文](DATA_FORMAT.md) | **English** · [README](../README.en.md)

Training requires three cache directory trees, each containing `train/` and `offline_valid/`. The three NPZ files for a sample must share the same filename, such as `process001_000.npz`. The reader matches files by name and retains only complete sets. When using real data, separately check the expected sample count and missing files; successful matching does not establish complete coverage.

| Cache | Key | Shape | Units / meaning |
|---|---|---|---|
| surface | x_mm | (6,H,W) | Historical hourly precipitation, mm |
| surface | y_mm | (3,H,W) | Precipitation for three user-defined forecast periods, mm |
| surface | y_valid | (3,H,W) | Valid-label indicator, 0/1 |
| surface | target_mask | (H,W) | Target-region mask, 0/1 |
| tp | tp_mm | (3,H,W) | NWP hourly precipitation aligned with the target lead times, mm |
| radar | radar_dbz | (3,H,W) | Three radar reflectivity frames, dBZ |

Precipitation is transformed with log1p before entering the network. Radar reflectivity is clipped to [0,80] and divided by 80. H and W should be divisible by 8. Training uses random crops; validation uses center crops. Evaluate real forecasts separately over the full grid: center-patch loss is not a full-region score.

Prepare data you are authorized to use in this format. Ensure that spatial and temporal coordinates, accumulated-precipitation differencing, missing-data masks, and operational availability times are correct. The interface itself does not verify these physical conditions. Sample IDs are not substitutes for weather-event identities.

The model outputs log1p precipitation and logits with shape `(B,3,5,H,W)`. To obtain precipitation in millimetres, apply `expm1` to the regression output and clip negative values to zero. Apply softmax along the class axis to obtain class probabilities. The two heads produce distinct outputs; do not assume that the predicted classes and regression amounts always agree.

**Missing modalities:** random modality dropout in the training entry point simulates missing inputs. During real inference, the caller must zero the corresponding data and availability flags together. `prepare_modalities(..., training=False)` assumes all modalities are available and does not automatically detect missing files.
