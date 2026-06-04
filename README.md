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

## Download the model zoo from: [https://drive.google.com/drive/folders/1nqrOKRf_jJXL8cQmASnFe7nOytilYCyL?usp=sharing](https://drive.google.com/drive/folders/1nqrOKRf_jJXL8cQmASnFe7nOytilYCyL?usp=sharing)

### Use this to run the code:

```
python run.py --root_path [DATASET PATH] --data_path [DATASET FILE] --model [MODEL] --data [DATASET NAME] --features [S,SM,M] --is_training 0 --pred_len [HORIZON SIZE] --enc_in [NUM VARIATES] --dec_in [NUM VARIATES] --c_out [NUM VARIATES] --itr [N] --model_params_json trained_models.json
```

If you found this repository useful, please consider citing: [https://arxiv.org/abs/2601.02094](https://arxiv.org/abs/2601.02094): Horizon Activation Mapping for Neural Networks in Time Series Forecasting
```
@misc{hans2026horizonactivationmappingneural,
      title={Horizon Activation Mapping for Neural Networks in Time Series Forecasting}, 
      author={Krupakar Hans and V A Kandappan},
      year={2026},
      eprint={2601.02094},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.02094}, 
}
```
