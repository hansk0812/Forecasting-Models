### Code patterns similar to Informer et al.

### This repository supports the following models for ETTm2:

###### NBEATS: https://github.com/Nixtla/neuralforecast
###### NHITS: https://github.com/Nixtla/neuralforecast
###### DLinear: https://github.com/cure-lab/LTSF-Linear
###### NLinear: https://github.com/cure-lab/LTSF-Linear
###### TiDE: https://github.com/google-research/google-research
###### FiLM: https://github.com/DAMO-DI-ML/NeurIPS2022-FiLM

###### SpaceTime: https://github.com/HazyResearch/spacetime

###### MultiResolutionDDPM: https://github.com/dlgudwn1219/mrDiff

###### Informer: https://github.com/MAZiqing/FEDformer
###### Autoformer: https://github.com/MAZiqing/FEDformer
###### FEDformer: https://github.com/MAZiqing/FEDformer
###### PatchTST: https://github.com/yuqinie98/PatchTST

###### Pyraformer: https://github.com/ant-research/Pyraformer
###### Triformer: https://github.com/razvanc92/triformer

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
