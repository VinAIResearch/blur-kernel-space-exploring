# Exploring Image Deblurring via Encoded Blur Kernel Space
This repository is the official pytorch implementation of the paper Exploring Image Deblurring via Encoded Blur Kernel Space.

![Blur kernel space](imgs/teaser.jpg)

## Dependencies and Installation
* Python >= 3.7
* Pytorch >= 1.4.0
* CUDA >= 10.0

## Usage
### Training
You can train your own model using the following command:
```
python train.py -opt path_to_yaml_file
```
where `path_to_yaml_file` is the path to yaml file that contain training configurations. You can find some default configurations in `options` folder. Checkpoints and logs will be saved in `../experiments/modelName`

### Testing
##### Data augmentation
To augment a given dataset, first, create an lmdb dataset using `scripts/create_lmdb.py`. Then using the following script:
```
python test_data_augmentation.py --target_H=256 --target_W=256 \\
                                 --model_path=experiments/pretrained/GOPRO_woVAE.pth \\
                                 --LQ_root=datasets/GOPRO/train_blur.lmdb \\
                                 --HQ_root=datasets/GOPRO/sharp_blur.lmdb \\
                                 --save_path=results/GOPRO_augmented \\
                                 --num_images=10000\\
                                 --yml_path=options/GOPRO/wsharp_woVAE.yml
```
`target_H` and `target_W` is the desired shape of the augmented images, `LQ_root` and `HQ_root` is the path of the lmdb dataset that was created before. `model_path` is the path of the trained model. `yml_path` is the path to the model configuration. Results will be saved in `save_path`.

![Data augmentation examples]
(imgs/augmentation.png)

##### Testing data augmentation with ground-truth
To be updated

##### Generate novel blur kernels
To be updated

![kernel generating examples]
()

##### Image Deblurring
To be updated

![Image deblurring examples]
(imgs/ddeblurring.png)

### Checkpoints
Pretrained models can be downloaded here.

## Important notes:
The training code is borrowed from EDVR project: https://github.com/xinntao/EDVR

## References
[1] Lehtinen, Jaakko, et al. "Noise2noise: Learning image restoration without clean data." arXiv preprint arXiv:1803.04189 (2018).
[2] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
[3] Batson, Joshua, and Loic Royer. "Noise2self: Blind denoising by self-supervision." arXiv preprint arXiv:1901.11365 (2019).
[4] Xie, Yaochen, Zhengyang Wang, and Shuiwang Ji. "Noise2Same: Optimizing A Self-Supervised Bound for Image Denoising." arXiv preprint arXiv:2010.11971 (2020).
