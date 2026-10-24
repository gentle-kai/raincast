# Method and Limitations

[简体中文](METHOD.md) | **English** · [README](../README.en.md)

## Background and residual

The shared U-Net receives 12 data channels and three modality-availability channels. Its regression output is the NWP log1p precipitation background plus a correction bounded by tanh. The ±0.35 bound applies in log1p space; it is neither a fixed correction in millimetres nor a constraint derived from dynamical equations. The regression head is initialized to zero, so its initial output equals the background.

## Classification and loss

The example uses intensity boundaries of 0.1, 5, 20, and 50 mm, yielding five left-inclusive, right-exclusive intervals, with the highest class unbounded above. The data provider defines the actual times associated with the three output periods. These boundaries are demonstration settings; changing them requires consistent changes to labels, loss functions, and documentation.

The joint loss is `0.30 * SmoothL1 + focal CE + 0.75 * Soft-TS`. The focal exponent is 2, and class weights are 0.15/1/5/25/100. Soft-TS is averaged over forecast periods and classes with observed events. It is a training surrogate, not an aggregate evaluation score.

Regression and classification are separate outputs and are not guaranteed to agree at every pixel. Clip regression precipitation to nonnegative values after converting it to millimetres.

## Sampling and missing inputs

Training can preferentially sample patches containing heavy precipitation; validation uses center crops by default. Real applications require separate full-grid evaluation. Random modality dropout does not automatically detect missing files; inference callers must explicitly handle inputs and availability channels.

## Reproducibility scope

Random synthetic data validates the software workflow only. It supports no conclusions about accuracy, generalization, or operational performance on real meteorological data. Training uses nondeterministic backend settings, and the crop random generator is not tied to the global seed, so bitwise reproducibility is not guaranteed.

For real applications, establish an NWP baseline under the same resource budget, then evaluate the incremental benefits of bounded residuals, multitask learning, and modality dropout separately. Hold out complete weather events for evaluation, and report results by forecast period and intensity, together with uncertainty estimates.
