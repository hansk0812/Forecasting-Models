### Code patterns similar to Informer et al.

### This repository supports the following models for ETTm2:

###### NBEATS
###### NHITS
###### DLinear
###### NLinear
###### TiDE
###### FiLM

###### Informer
###### Autoformer
###### FEDformer
###### PatchTST

###### Triformer

## Download the model zoo from: [https://bit.ly/LHFModelZoo](https://bit.ly/LHFModelZoo)

### Use this to run the code:

```
python run.py --data ETTm2 --root_path [ETT-small dir path] --data_path [ETTXX.csv] --model [MODEL] --is_training 0 --load_from_zoo --pred_len [96,192,336,720] --features [S,SM,M]
```

If you found this repository useful, please cite: A Review of the Long Horizon Forecasting Problem in Time Series Analysis
