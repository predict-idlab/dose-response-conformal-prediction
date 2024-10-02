# Conformal Prediction for Dose-Response Models

> This library provides a conformal prediction framework for dose-response models and for uncertainty quantification in counterfactual explanation of continuous features

## Installation ⚙️

_WORK IN PROGRESS_

## Usage 🛠

The conformal prediction for dose-response models builds upon the Crepes and weighted crepes framework.

```py
from WCDRF.base import *

X, T, y = ...  # your continuous treatment dataset

X_train, X_cal = ... # your calibration and train split for covariates X
T_train, T_cal = ... # your calibration and train split for treatment T
y_train, y_cal = ... # your calibration and train split for outcome y

X_target = ... # your target sample or target dataset

# You can specify any regression model for either propensity or outcome model
propensity_model = ...
outcome_model = ... 

# Specify the DRFWrapRegressor which is an extension of the Wrap Regressor from Weighted Crepes for Dose-Response Functions (DRF)
dose_response_wrapper = DRFWrapRegressor(propensity_model, outcome_model)
dose_response_wrapper.fit([X_train, T_train], y_train)

# Calibrate and fit the propensity estimator which uses Conformal Predictive Systems under the hood
dose_response_wrapper.fit_propensity(X_train, T_train)
dose_response_wrapper.calibrate_propensity(X_cal, T_cal)

# If you are planning to perform many calibrations on the same calibration set, it is computationally better to perform a prepare_calibration, it uses a multiplier of 0.2 by default
dose_response_wrapper.prepare_calibration(X_cal, T_cal)

# Define a treatment vector t0_vector with length equal to X_train.
target_dataset = ... # concat X_target with t0_vector

# The options for calibrate are: CPS (default=False, to use conformal predictive systems), use_propensity (default=True, to use the propensity weights), and local_conditional_mode (default=True, to use the local mode)
dose_response_wrapper.calibrate(target_dataset, y_cal, T_cal, target_treatment = t0)

# If you want the prediction intervals for multiple coverages, e.g. 50%, 80%, 90%, and 95% use:
prediction_intervals_matrix = dose_response_wrapper.predict_multi_int(
                                X = target_dataset,
                                y_min = np.min(y_train) - np.abs(np.min(y_train)),
                                y_max = np.max(y_train) + np.abs(np.max(y_train)),
                                confidence_range = [0.5, 0.8, 0.9, 0.95]
                                )
# prediction_intervals_matrix has dimensions: [coverage_idx, target_dataset_idx, 2], with [coverage_idx, target_dataset_idx, 0] being the lower bound and [coverage_idx, target_dataset_idx, 1] the upper bound

# Otherwise simply use
prediction_intervals = dose_response_wrapper.predict_int(
                                X = target_dataset,
                                y_min = np.min(y_train) - np.abs(np.min(y_train)),
                                y_max = np.max(y_train) + np.abs(np.max(y_train)),
                                confidence = 0.9
                                )
# prediction_intervals has dimensions: [target_dataset_idx, 2], with [target_dataset_idx, 0] being the lower bound and [target_dataset_idx, 1] the upper bound
```
the [WCDRF_experiments](https://github.com/predict-idlab/conformal_prediction_dose_response/blob/main/WCDRF_experiments.ipynb) notebook contains the experiments of the paper and shows how to apply the method. 

Additionally, to generate synthetic data from any of the three setups and scenarios the following code can be utilized, using code from synthetic_data_generation.py:

```py
from WCDRF.synthetic_data_generation import *

EXPERIMENT_SOURCE = 2
EXPERIMENT_SCENARIO = 1

synthetic_generator = synthetic_data_generator(source = EXPERIMENT_SOURCE, scenario = EXPERIMENT_SCENARIO)
synthetic_DRF_df = synthetic_generator.generate_synthetic_DRF_data(N = 5000)

# If you would need the true propensity distribution
OraclePropensityWrapperObject = OraclePropensityWrapper(synthetic_generator)

features = list(synthetic_DRF_df.columns.values[:-2])
treatment = "W"
outcome = "Y"
```

## Features ✨

* Continuous treatment uncertainty quantification
* Modifiable weights
* Model-agnostic
* Synthetic data generators
 
## Referencing our package :memo:

If you use this code in a scientific work, we would highly appreciate citing us as:

```bibtex
@misc{verhaeghe2024conformalpredictiondoseresponsemodels,
      title={Conformal Prediction for Dose-Response Models with Continuous Treatments}, 
      author={Jarne Verhaeghe and Jef Jonkers and Sofie Van Hoecke},
      year={2024},
      eprint={2409.20412},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.20412}, 
}
```
