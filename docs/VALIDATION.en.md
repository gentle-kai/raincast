# Software Validation

[简体中文](VALIDATION.md) | **English** · [README](../README.en.md)

**Results:** all three model-contract tests passed, covering bounded residuals, modality dropout, masked loss, and finite gradients. Random cache generation, one CPU training epoch, checkpoint saving, and checkpoint loading for inference also passed.

**Environment:** Windows, Python 3.12.14, NumPy 2.3.5, and PyTorch 2.8.0+cpu. Existing local dependencies were used; a fresh virtual environment and GPU execution have not been validated.

**Synthetic dataset:** four training samples and two validation samples on a 32×32 grid. Inference produces precipitation amounts with shape `(1,3,32,32)` and class probabilities with shape `(1,3,5,32,32)`.

GitHub Actions has also successfully run the automated checks; see the [verified run](https://github.com/gentle-kai/raincast/actions/runs/40076069460). Random data confirms that the software runs, not that it has forecasting skill.
