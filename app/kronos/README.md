# Kronos forecasting adapter

`KronosForecastAdapter` is the application boundary around the repository's
existing `model.Kronos`, `model.KronosTokenizer`, and `model.KronosPredictor`.
It does not modify or duplicate upstream model code.

## Integration requirements

- Install the root `requirements.txt` dependencies, including PyTorch.
- Load checkpoints with `KronosForecastAdapter.from_pretrained(...)`; this
  requires access to the configured local or Hugging Face model identifiers.
- Pass a DataFrame containing `timestamp`, `open`, `high`, `low`, `close`, and
  `volume`. An `amount` column is optional and defaults to `volume * close`.
- Configure `timeframe` with a pandas-compatible offset (for example, `5min`
  or `1D`). If future timestamps are not supplied, the adapter generates them.
- The default device is CPU. GPU/MPS devices require the corresponding PyTorch
  runtime and can be checked with `validate_device=True`.

The result contains deterministic forecast points and explicit `None` values
for `confidence` and `forecast_range`: the upstream predictor does not provide
calibrated uncertainty or prediction intervals.
