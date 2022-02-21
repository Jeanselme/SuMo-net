# SuMo-net
This repository is an implementation of [Survival Regression with Proper Scoring Rules and Monotonic Neural Networks](https://arxiv.org/abs/2103.14755).
This paper approaches models the cumulative hazard function using positive and monotone neural networks.

## How to use the model ?
To use the model, one needs to execute:
```python
from sumo import SuMo
model = SuMo()
model.fit(x, t, e)
model.predict_risk(x)
```

A full example with analysis is provided in `examples/SuMo on SUPPORT Dataset.ipynb`.

# Setup
## Structure
We followed the same structure than the [DeepSurvivalMachines](https://github.com/autonlab/DeepSurvivalMachines) repository with the model in `sumo/` - only the api should be used to test the model. Examples are provided in `examples/`. 

## Requirements
The model relies on `DeepSurvivalMachines`, `pytorch`, `numpy` and `tqdm`.  
To run the set of experiments `pycox`, `lifelines`, `pysurvival` are necessary.