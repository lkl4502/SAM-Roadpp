
## Installation
You need the following:
- an Nvidia GPU with latest CUDA and driver.
- the latest pytorch.
- pytorch lightning.
- wandb.
- Go, just for the APLS metric (we should really re-write this with pure python when time allows).
- and pip install whatever is missing.


## Getting Started

### SAM Preparation
Download the ViT-B checkpoint from the official SAM directory. Put it under:  
-sam_road  
--sam_ckpts  
---sam_vit_b_01ec64.pth  

### Data Preparation
Refer to the instructions in the sam_road repo to download City-scale and SpaceNet datasets.
Put them in the main directory, structure like:  
-sam_road++  
--cityscale  
---20cities  
--spacenet  
---RGB_1.0_meter  

and run python generate_labes.py under both dirs.

for Global-scale, refer to cityscale for data preparation

### Training
City-scale dataset:  
python train.py --config=config/toponet_vitb_512_cityscale.yaml  

Glbale-scale dataset:  
python train.py --config=config/toponet_vitb_512_globalscale.yaml 

or

python train.py --config=config/toponet_vitb_256_globalscale.yaml 

SpaceNet dataset:  
python train.py --config=config/toponet_vitb_256_spacenet.yaml  

You can find the checkpoints under lightning_logs dir.

### Inference
python inferencer.py --config=path_to_the_same_config_for_training --checkpoint=path_to_ckpt  
This saves the inference results and visualizations.

### Test
Go to cityscale_metrics（both for cityscale and globalscale） or spacenet_metrics, and run  
bash eval_schedule.bash  

Check that script for details. It runs both APLS and TOPO and stores scores to your output dir.



## Acknowledgement
We sincerely appreciate the authors of the following codebases which made this project possible:
- Segment Anything Model  
- Samroad 
- SAMed  
- Detectron2  

## TODO List
- [√] Basic instructions
- [√] Organize configs
- [√] Add dependency list
- [ ] Add demos
- [ ] Add trained checkpoints



