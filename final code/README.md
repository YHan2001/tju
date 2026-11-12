# Terrain-adaptive PV forecasting and dispatch optimization

This folder accompanies the manuscript “Terrain-adaptive forecasting overcomes geographical barriers to global solar energy deployment.” It contains code for forecasting photovoltaic (PV) power and evaluating how forecast quality affects operating cost, PV curtailment, and balancing purchases in simulated dispatch scenarios.

## Files

| File | Purpose |
| --- | --- |
| [`PreModel_LSTM-CNN-LSTM-GRU.ipynb`](./PreModel_LSTM-CNN-LSTM-GRU.ipynb) | TensorFlow/Keras workflow for PV power forecasting. The study compares LSTM, GRU, and CNN–LSTM. The notebook is currently configured for GRU; the model cell includes commented LSTM and Conv1D alternatives. The CNN–LSTM architecture uses a Conv1D feature extraction layer followed by a standard LSTM. |
| [`Multi-O-PV.py`](./Multi-O-PV.py) | PuLP implementation of an ε-constraint dispatch analysis with 24-, 48-, and 72-hour planning horizons, battery storage, and multiple load scenarios. |

The folder contains the two model files. The forecasting notebook expects prepared site-level inputs, and the dispatch script expects exported forecast and observation series.

## Forecasting

1. Open the notebook in Jupyter with an environment that provides TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, and seaborn. The notebook records Python 3.8.13 as its original kernel version. It includes a legacy Keras scikit-learn wrapper import; if that import is unavailable, omit the unused wrapper import for the current GRU run.
2. Change the input path from `./era5gfs_72h.csv` to your prepared CSV. Use a `time` column and place the observed PV power target in the last column. Align historical PV power, historical meteorology, and forecast meteorology to the intended forecast times before running the notebook. Prepare separate predictor configurations for the baseline and OPT runs using the screening procedure in the manuscript.
3. Check the chronological split, model configuration, and file paths before running. The notebook example uses days 1–21 of each month for training and later days for testing. It uses 72 input and 72 output time steps, which correspond to 72 hours when the input is hourly. For an independent evaluation, use a separate validation set for early stopping or model selection; the example training call passes the test partition as validation.
4. Create the output directories or update the notebook's save paths. The notebook exports observations and predictions on their original scale and reports R², MAE, MSE, and RMSE.

## Dispatch optimization

Install NumPy, pandas, Matplotlib, and PuLP, and make a compatible linear programming solver such as CBC available to PuLP. Run the following command from this folder after setting the input paths in the script:

```bash
python Multi-O-PV.py
```

- Set `GOOD_PRED_PATH` to the forecast CSV produced with OPT and `BAD_PRED_PATH` to the matching baseline CSV when comparing those two configurations. Keep the site, time span, power units, and ordering consistent across files.
- The script reads each CSV with `header=None`. With one-based row numbering, it reads observed PV power from row 2 of the OPT file and predictions from row 3 of each file. Row 1 can contain the column headers. Export with `index=False` to avoid an extra index column. Flatten consecutive, overlapping 72-step forecast windows in chronological order, with one time step between successive window starts. The script combines those windows into an hourly series using `WINDOW_SIZE` and `KEEP_FIRST`.
- By default, the script uses synthetic load profiles. To use an external load file, set `LOAD_PATH` and prepare the load in the same rolling-window format. The script solves at each planning horizon and evaluates the first 24 hours of each schedule against observed PV power.
- The scheduling objective includes thermal generation and battery cycling costs. The evaluation reports thermal generation cost plus balancing purchase cost, realized curtailment, and balancing purchases. The script scans curtailment limits for each forecast and re-solves without the cap if the constrained solve fails. Treat the resulting plots as scenario comparisons and check cap feasibility if interpreting them as a Pareto frontier.
- The default output folder, `results_scenarios_v3/`, contains per-scenario CSVs and cost–curtailment plots, along with `summary_all_scenarios.csv`.

## Public data sources

| Data | Source |
| --- | --- |
| ERA5 historical meteorology | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) |
| GFS forecast products | [NOAA NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) |
| Historical meteorology for the international evaluation | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) |
| Archived GFS forecasts for Tozeur | [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) |
| Archived GFS forecasts for Faro | [NSF NCAR Geoscience Data Exchange](https://doi.org/10.5065/D65D8PWK) |
| Sun Metals Solar Farm, Australia | [AEMO generation and load data](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-management-system-mms-data/generation-and-load), `DISPATCH_UNIT_SCADA`, DUID `SMCSF1` |
| Faro, Portugal | [EDP Open Data](https://edp.com/en/innovation/data), Sunlab Faro PV 2016 |
| Bolzano, Italy | [TRUST-PV monitoring dataset](https://doi.org/10.5281/zenodo.14766289) |
| Solar Esperanza, Chile | [Electricity output for grid-connected PV installations in Chile](https://doi.org/10.5281/zenodo.3939047), Solar Esperanza station record |
| SR_CO, United States | [PVDAQ Public Datasets](https://doi.org/10.25984/1846021), system `9068` |
| Tozeur, Tunisia | [PV and Weather Datasets from Demosites in Oslo, Touzer, and Sevilla](https://doi.org/10.5281/zenodo.10423357), Tozeur records |

For the study-specific inputs, station periods, and experimental settings, see the Materials and Methods, Supplementary Materials, and the data availability statement in the manuscript.
