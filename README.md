# Smart Drying with Timestral

A research prototype for **rolling time-series forecasting and decision support in drying processes**, based on [Timestral by Mistral AI](https://github.com/mistralai/timestral).

The idea is simple: instead of running a dryer mainly according to a fixed recipe and elapsed time, use the measurements collected during the process to continuously estimate **when the desired drying target will be reached**.

The current repository uses hot-air drying of potato samples as an experimental case study. The workflow itself is not specific to potatoes and is intended as a general framework for drying processes represented as time series.

## Why Timestral?

Drying is a dynamic process. Temperature, air conditions, product properties and pretreatments can change the shape and duration of the drying curve.

Classical process control may rely on predefined operating conditions, fixed processing times and operator experience:

**Conventional drying**

`process settings → elapsed time → operator assessment → stop`

This project explores a different approach:

**Smart drying**

`process measurements → rolling forecast → estimated time-to-target → updated decision → stop`

At regular intervals, all measurements collected so far are provided to Timestral. The observed window therefore becomes progressively longer, while the remaining interval to be forecast becomes shorter.

The prediction evolves together with the process.

In this example, the predicted time required to reach the selected moisture target is repeatedly updated as new measurements become available. Early predictions are more uncertain; the estimate is progressively refined as the process develops.

## Why a time-series foundation model?

Timestral is a family of foundation models developed specifically for probabilistic time-series forecasting.

This makes it interesting for smart drying because it can provide:

- **zero-shot forecasting**, without training a dedicated model from scratch for every experiment;

- **rolling predictions**, updated as new process measurements become available;

- **probabilistic forecasts**, useful for describing prediction uncertainty;

- support for **past and known-future covariates**, enabling future integration of process variables;

- compact model configurations with CPU inference support.

The last point is particularly relevant for process engineering. A forecasting layer that does not depend on expensive computing infrastructure could potentially be integrated directly into a dryer, an industrial controller or a low-cost edge computer.

Embedded deployment is not yet benchmarked in this project and should therefore be considered a development objective rather than a demonstrated result.

## From prediction to decision support

The objective is not simply to obtain a low forecasting error.

The useful question during drying is:

**How much longer should the process continue to reach the desired target?**

The current workflow evaluates:

- predicted time of target achievement;

- remaining process time;

- error in the predicted stopping time;

- actual moisture content corresponding to the predicted stopping time;

- evolution of forecast uncertainty;

- forecasting performance as additional process history becomes available.

This makes the output directly interpretable for process supervision.

The operator still defines the technological target and retains control of the process. Forecasting provides quantitative evidence to support that decision.

This human-centred use of AI is consistent with the principles of **Industry 5.0**: the goal is not simply to automate the dryer, but to combine process measurements, predictive intelligence and operator decisions.

## Current case study

The experimental dataset currently used for validation consists of hot-air drying curves obtained from potato samples under different processing conditions.

Moisture content on a dry basis (MCdb) is currently used as the forecasting variable, and several semi-drying targets are evaluated.

The case study is used to test the methodology, not to define its application boundary.

The same approach could be extended to other foods, biomaterials or drying systems.

Future process information could include:

- air temperature;

- relative humidity;

- air velocity;

- product temperature;

- geometrical information or shrinkage obtained from imaging systems.

## Workflow

The active pipeline is intentionally small:

- `scripts/01_timestral_rolling.py`  
  Rolling forecast for one process trajectory and one selected target.

- `scripts/02_timestral_validate_all.py`  
  Validation across multiple trajectories and multiple drying targets.

- `scripts/03_timestral_plots.py`  
  Generation of summary statistics and diagnostic figures.

- `scripts/timestral_utils.py`  
  Shared Timestral inference and evaluation functions.

The scripts are numbered according to their role in the workflow.

Experimental data and generated results are kept local and are not distributed with the public repository.

## Timestral

This project uses the **Timestral** time-series foundation model developed by Mistral AI.

Official repository: [mistralai/timestral](https://github.com/mistralai/timestral)

The current implementation uses zero-shot inference: the Timestral model weights are not trained or updated using the drying experiments.

## Status

This repository is a **research prototype**.

It currently demonstrates and evaluates a methodological framework for foundation-model-assisted drying decision support.

It should not yet be considered a validated industrial control system.

Planned developments include:

- comparison of different Timestral model sizes;

- evaluation on additional drying processes and materials;

- integration of physical and sensor-derived covariates;

- benchmarking on low-cost edge hardware;

- assessment of computational requirements for embedded deployment.
