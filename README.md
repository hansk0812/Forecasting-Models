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

###### Pyraformer
###### Triformer

## Download the model zoo from: [https://bit.ly/LHFModelZoo](https://bit.ly/LHFModelZoo)

### Use this to run the code:

```
python run.py --root_path [ETT-small DIR PATH] --data_path ETTm2.csv --model [MODEL] --data ETTm2 --features [S,SM,M] --is_training 0 --pred_len [96,192,336,720] --enc_in [1,7] --dec_in [1,7] --c_out [1,7] --itr [N] --model_params_json trained_models.json
```

If you found this repository useful, please cite: [https://arxiv.org/abs/2506.12809](https://arxiv.org/abs/2506.12809): A Review of the Long Horizon Forecasting Problem in Time Series Analysis
```
@misc{krupakar2025reviewlonghorizonforecasting,
      title={A Review of the Long Horizon Forecasting Problem in Time Series Analysis}, 
      author={Hans Krupakar and Kandappan V A},
      year={2025},
      eprint={2506.12809},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.12809}, 
}
```
