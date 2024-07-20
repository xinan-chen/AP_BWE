# Towards High-Quality and Efficient Speech Bandwidth Extension with Parallel Amplitude and Phase Prediction
This is the unofficial implementation of [Towards High-Quality and Efficient Speech Bandwidth Extension with Parallel Amplitude and Phase Prediction](https://arxiv.org/abs/2401.06387). 
![](apbwe.png)


## Training:

first prepare data

To train this model, run his code:

```
python train.py --gpu_avail [gpu_ids] --batch_size [batch] --init_lr [initial learning rate] --epochs [number of epochs of training] or --steps [number of steps where training stops] --data_dir [dir of VCTK dataset]
```
or you may change default options in train.py


## Notes:
* refercode folder contains reference code from https://github.com/jik876/hifi-gan & https://github.com/facebookresearch/ConvNeXt & https://github.com/huyanxin/phasen
* runs folder contains tensorboard logs
* AudioSamples folder contains samples for comparison, including spectra

## Citations:
```
@article{lu2024towards,
  title={Towards high-quality and efficient speech bandwidth extension with parallel amplitude and phase prediction},
  author={Lu, Ye-Xin and Ai, Yang and Du, Hui-Peng and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2401.06387},
  year={2024}
}

```

