<div align="center">

<h1>Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method</h1>


<div>
    <a href='https://github.com/Pancool303' target='_blank'>Pan Yin</a><sup>1*</sup>&emsp;
    <a href='https://likyoo.github.io/' target='_blank'>Kaiyu Li</a><sup>1*</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/caoxiangyong' target='_blank'>Xiangyong Cao</a><sup>✉1</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=1SHd5ygAAAAJ' target='_blank'>Jing Yao</a><sup>2</sup>&emsp;
    <a href='https://web.xidian.edu.cn/leiliusee' target='_blank'>Lei Liu</a><sup>3</sup>&emsp;
    <a href='https://web.xidian.edu.cn/xrbai' target='_blank'>Xueru Bai</a><sup>3</sup>&emsp;
    <a href='https://faculty.xidian.edu.cn/ZF3' target='_blank'>Feng Zhou</a><sup>3</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/dymeng' target='_blank'>Deyu Meng</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Xi'an Jiaotong University&emsp;
    <sup>2</sup>Chinese Academy of Sciences&emsp;
    <sup>3</sup>Xidian University&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://github.com/earth-insights/samroadplus" target='_blank'>[Project]</a> • <a href="https://arxiv.org/abs/2411.16733" target='_blank'>[arXiv]</a> • • <a href="https://pan.baidu.com/s/1_bWmY3W8GVvizX3W2jLrWg?pwd=a9p7" target='_blank'>[Dataset]</a>
    </h4>
</div>
<img src="https://github.com/earth-insights/samroadplus/blob/main/img/overview.PNG" width="100%"/>
Overview of our proposed SAM-Road++. The <font color="red">red line</font> indicates training only and the  <font color="blue"> blue line</font> indicates inference only.

</div>

## Abstract
> *Recently, road graph extraction has garnered increasing attention due to its crucial role in autonomous driving, navigation, etc. However, accurately and efficiently extracting road graphs remains a persistent challenge, primarily due to the severe scarcity of labeled data. To address this limitation, we collect a global-scale satellite road graph extraction dataset, i.e. Global-Scale dataset. Specifically, the Global-Scale dataset is ~20× larger than the largest existing public road extraction dataset and spans over 13,800 km^2 globally. Additionally, we develop a novel road graph extraction model, i.e. SAM-Road++, which adopts a node-guided resampling method to alleviate the mismatch issue between training and inference in SAM-Road, a pioneering state-of-the-art road graph extraction model. Furthermore, we propose a simple yet effective "extended-line" strategy in SAM-Road++ to mitigate the occlusion issue on the road. Extensive experiments demonstrate the validity of the collected Global-Scale dataset and the proposed SAM-Road++ method, particularly highlighting its superior predictive power in unseen regions.*

## Installation
You need the following:
- an Nvidia GPU with latest CUDA and driver.
- the latest pytorch.
- pytorch lightning.
- wandb.
- Go, just for the APLS metric.
- and pip install whatever is missing.

## Getting Started

### SAM Preparation
Download the ViT-B checkpoint from the official SAM directory. Put it under:  
```
-sam_road++  
--sam_ckpts  
---sam_vit_b_01ec64.pth  
```

### Data Preparation
Refer to the instructions in the sam_road repo to download City-scale and SpaceNet datasets.
Put them in the main directory, structure like:  
```
-sam_road++  
--cityscale  
---20cities  
--spacenet  
---RGB_1.0_meter  
```
And run python generate_labes.py under both dirs.

For Global-scale, refer to cityscale for data preparation. And you can download the Global-scale datasets from this [link](https://pan.baidu.com/s/1_bWmY3W8GVvizX3W2jLrWg?pwd=a9p7).
### Training
City-scale dataset:  

```
python train.py --config=config/toponet_vitb_512_cityscale.yaml  
```

Glbale-scale dataset:
```
python train.py --config=config/toponet_vitb_512_globalscale.yaml
```
or
```
python train.py --config=config/toponet_vitb_256_globalscale.yaml
```


SpaceNet dataset:
```
python train.py --config=config/toponet_vitb_256_spacenet.yaml 
```

You can find the checkpoints under lightning_logs dir.

### Inference
```
python inferencer.py 
--config=path_to_the_same_config_for_training--checkpoint=path_to_ckpt  
```

### Test
For APLS and TOPO metrics, please move to [Sat2Graph](https://github.com/songtaohe/Sat2Graph). It is worth mentioning that the metrics used to test our Global-scale datasets are the same as those used for the Cityscale datasets.

## Demos
<img src="https://github.com/earth-insights/samroadplus/blob/main/img/vis.PNG" width="100%"/>
Visual road network graph prediction based on SAM-Road++ and two currently advanced methods.




## Citation

```
@article{yin2024satelliteimageroadgraph,
  title={Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method},
  author={Yin, Pan and Li, Kaiyu and Cao, Xiangyong and Yao, Jing and Liu, Lei and Bai, Xueru and Zhou, Feng and Meng, Deyu},
  journal={arXiv preprint arXiv:2411.16733},
  year={2024}
}
```

## Acknowledgement
We sincerely appreciate the authors of the following codebases which made this project possible:
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)  
- [SAM_Road](https://github.com/htcr/sam_road) 
- [Sat2Graph](https://github.com/songtaohe/Sat2Graph)
- [SAMed](https://github.com/hitachinsk/SAMed)  
- [Detectron2](https://github.com/facebookresearch/detectron2)  

## TODO List
- [x] Basic instructions
- [ ] Organize configs
- [x] Add dependency list
- [ ] Add demos
- [ ] Add trained checkpoints
