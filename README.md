# Neural Camera Simulators


**This repository includes official codes for "[Neural Camera Simulators (CVPR2021)](https://arxiv.org/abs/2104.05237)".** 

> **Neural Camera Simulators** <br>
>  Hao Ouyang*, Zifan Shi*, Chenyang Lei, Ka Lung Law, Qifeng Chen (* indicates joint first authors)<br>
>  HKUST <br>

[[Paper](https://arxiv.org/abs/2104.05237)] 
[[Datasets(Coming soon)](https://aaaa.github.io/TBA)] 



## Installation
Clone this repo.
```bash
git clone https://github.com/ken-ouyang/neural_image_simulator.git
cd neural_image_simulator/
```

We have tested our code on Ubuntu 18.04 LTS with PyTorch 1.3.0 and CUDA 10.1. Please install dependencies by
```bash
conda env create -f environment.yml
```

## Preparing datasets
We provide two datasets for training and test: [[Nikon](https://aaaa.github.io/TBA)] and [[Canon](https://aaaa.github.io/TBA)]. The data can be preprocessed with the command:
```bash
python preprocess/preprocess_nikon.py --input_dir the_directory_of_the_dataset --output_dir the_directory_to_save_the_preprocessed_data --image_size 512
OR
python preprocess/preprocess_canon.py --input_dir the_directory_of_the_dataset --output_dir the_directory_to_save_the_preprocessed_data --image_size 512
```
The preprocessed data can also be downloaded with the link [[Nikon](https://aaaa.github.io/TBA)] and [[Canon](https://aaaa.github.io/TBA)]. The preprocessed dataset can be put into the folder `./ProcessedData/Nikon/` or `/ProcessedData/Canon/`

## Training networks
The training arguments are specified in a json file. To train the model, run with the following code
```bash
python train.py --config config/config_train.json
```
The checkpoints will be saved into `./exp/{exp_name}/`. 
When training the noise module, set `unet_training` in the json file to be `true`. Other times it will be false. `aperture` is `true` when training the aperture module while other times it is `false`.

## Demo
Download the pretrained demo [[checkpoints](https://aaaa.github.io/TBA)] and put them under `./exp/demo/`. Then, run the command
```bash
python demo_simulation.py --config config/config_demo.json
```
The simulated results are available under `./exp/{exp_name}`

## Citation

```
@inproceedings{ouyang2021neural,
  title = {Neural Camera Simulators},
  author = {Ouyang, Hao and Shi, Zifan and Lei, Chenyang and Law, Ka Lung and Chen, Qifeng},
  booktitle = {CVPR},
  year = {2021}
}
```
## Acknowledgement
Part of the codes benefit from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) and [pyexiftool](https://github.com/smarnach/pyexiftool). 