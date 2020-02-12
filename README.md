# SketchyScene: Richly-Annotated Scene Sketches

This repository hosts the datasets and the code for training the model. Please refer to our ECCV paper for more information: ["SketchyScene: Richly-Annotated Scene Sketches
"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Changqing_Zou_SketchyScene_Richly-Annotated_Scene_ECCV_2018_paper.pdf)

## Content
- [Dataset](#dataset)
- [USketch](#usketch)
- [DuCrawler](#ducrawler)
- [Semantic Segmentation [TensorFlow]](#semantic-segmentation) | [[PyTorch]](https://github.com/MarkMoHR/SketchyScene-pytorch)
- [Instance Segmentation [TensorFlow]](#instance-segmentation) | [[PyTorch]](https://github.com/MarkMoHR/SketchyScene-pytorch)
- [Citation](#citation)

## Dataset

Our datasets consist of three part:
* *SketchyScene-7k* 
    * *SketchyScene-7k* contains 7265 crowdsourced sketchy scenes (train 5617 + val 535 + test 1113).
* *SketchyScene-components* 
    * *SketchyScene-components* contains the single object sketches and their composing order in the scene sketches.
* *SketchyScene-Selected3*
	* *SketchyScene-Selected3* contains 3 manually selected, synthesized sketchy scenes based on the crowdsourced template, resulting 21,795 sketchy scenes. 
* *SketchyScene-Synthesized30* 
    * *SketchyScene-Synthesized30* contains 30 synthesized examples for each template, resulting additional 217,950 sketchy scenes.
	
### UMIACS Hosting (7-zip)

* [SketchyScene-7k](https://obj.umiacs.umd.edu/private_datasets/SketchyScene-7k.7z) (750.3 MB)
* [SketchyScene-Selected3](https://obj.umiacs.umd.edu/private_datasets/SketchyScene-Selected3.7z) (1.6 GB)
* [SketchyScene-Synthesized30](https://obj.umiacs.umd.edu/private_datasets/SketchyScene-Synthesized30.7z) (16.1 GB)

### Google Drive Hosting (tar or zip)
* [SketchyScene-7k](https://drive.google.com/open?id=1m1fac2XIZVAGu_ByE6BtwxGSLytZspO-)
* [SketchyScene-components](https://drive.google.com/drive/folders/1Nd6XWSi8UQfIVpuesVpFO4VY0NgAx1-K?usp=sharing)
* [SketchyScene-Selected3](https://drive.google.com/drive/folders/1x7DiyTlpEFb_fydOyjL48wnvREQO1u1d?usp=sharing)
* [SketchyScene-Synthesized30](https://drive.google.com/drive/folders/15TWNXFOKoB0dKkOaDofFgLJ_9JuxgStm?usp=sharing)


## USketch

**USketch** is a web-driven tool for crowdsourcing the **SketchyScene** dataset. It is open-sourced at https://github.com/ruofeidu/USketch. You can also clone and update the submodules of this repo to acquire the full source code:

You must run two commands:  to initialize your local configuration file, and git submodule update to fetch all the data from that project and check out the appropriate commit listed in your superproject:

```
git clone git@github.com:SketchyScene/SketchyScene.git
git submodule init
git submodule update
```

A live demo is presented at http://go.duruofei.com/skenew/?task=1.

![USketch Interface](figures/USketch.jpg "Interface and work flow of USketch for crowdsourcing the dataset. See areas of function buttons (upper left), component display (lower left), and canvas (right). ")


## DuCrawler

In preparation for the reference data in **USketch**, we have developed a custom image crawler to acquire 7,500 cartoon scenes and 9,290 sketchy objects in 45 categories for academic use. Due to the length limitation, we explain the detailed crawling process and source code in the supplementary materials.

![Object instance frequency](figures/DuCrawler.jpg "Object instance frequency for each category.")

We selected 45 categories for our dataset, including objects and stuff classes. Specifically, we first considered several common scenes (e.g., garden, farm, dinning room, and park) and extracted 100 objects/stuff classes from them as raw candidates. Then we defined three super-classes, i.e. Weather, Object, and Field (Environment), and assigned the candidates into each super-class. Finally, we selected 45 from them by considering their combinations and commonness in real life. 

Instead of asking workers to draw each object, we provided them with plenty of object sketches (each object candidate is also refer to a ``component") as candidates. In order to have enough variations in the object appearance in terms of pose and appearance, we searched and downloaded around 1,500 components for each category. 

## Semantic Segmentation

The code under `Semantic_Segmentation` is for the semantic segmentation experiments of our SketchyScene dataset.

### Requirements

- Python 3
- Tensorflow (>= 1.3.0)
- Numpy
- PIL (Pillow version = 2.3.0)
- [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

### Preparations

- Download the whole dataset and place them under `data` directory following its instructions.
- Generate the ImageNet pre-trained "ResNet-101" model in TensorFlow version for initial training and place it under the `resnet_pretrained_model` directory. This can be obtained following the instructions in [chenxi116/TF-resnet](https://github.com/chenxi116/TF-resnet#example-usage). For convenience, you can download our converted model [here](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c?usp=sharing). 

### Training

After the preparations, run:

```
python3 segment_main.py --mode=train
```

Also, you can modify the training parameters in `configs.py`


### Evaluation

Evaluation can be done with `val` and `test` dataset. Make sure that your trained tfmodel is under the directory `Semantic_Segmentation/outputs/snapshot`. [DenseCRF](https://github.com/lucasb-eyer/pydensecrf) can be used to improve the segmentation performance as a post-processing skill.

For evaluation under `val`/`test` dataset without DenseCRF, run:
```
python3 segment_main.py --mode='val' --dcrf=0
python3 segment_main.py --mode='test' --dcrf=0
```

- DenseCRF is used if setting `--dcrf=1`

Our trained semantic segmentation model can be download [here](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c?usp=sharing).


### Inference

You can obtain a semantic segmentation output during inference. Inference can be done with `val` and `test` dataset.

For inference with the 2nd image in `val` dataset without DenseCRF, which the background is white, run:

```
python3 segment_main.py --mode='inference' --infer_dataset='val' --image_id=2 --black_bg=0  --dcrf=0
```

- Inference under `test` dataset if setting `--infer_dataset='test'`
- Try other image if setting `--image_id` to other number
- The background is black if setting `--black_bg=1`. Otherwise, it is white.
- DenseCRF is used if setting `--dcrf=1`

Also, you can try [our trained model](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c?usp=sharing).


### Visualization

You can visualize the ground-truth semantic results with the `.mat` data using `Semantic_Segmentation/tools/semantic_visualize.py`. **Note** that the data should be correctly arranged following the instructions under `data` directory.

For visualization with the 1st/2nd image in `train` dataset, run:

```
python3 semantic_visualize.py --dataset='train' --image_id=1 --black_bg=1
python3 semantic_visualize.py --dataset='train' --image_id=2 --black_bg=0
```

- Visualization under `val`/`test` dataset if setting `--dataset='val'` or `--dataset='test'`
- Try other image if setting `--image_id` to other number
- The background is black if setting `--black_bg=1` and white if `--black_bg=0`.


## Instance Segmentation

The code under `Instance_Segmentation` is for the instance segmentation experiments of our SketchyScene dataset.

### Requirements

- Python 3
- Tensorflow (>= 1.3.0)
- Keras 2.0.8
- Other common packages listed in requirements.txt

### Preparations

- Download the whole dataset and place them under `data` directory following its instructions.
- Download the coco/imagenet pre-trained model following the instructions under `Instance_Segmentation/pretrained_model`. 

### Training

After the preparations, run:

```
python3 segment_train.py
```
or
```
python3 segment_train.py --init_model='coco'
```

- Choose the initial pre-trained model from ['coco', 'imagenet', 'last'] at `--init_model`. Train from the fresh start if not specified. 'last' denotes your lastly trained model.
- Other settings can be modified at `SketchTrainConfig` in this file.

### Evaluation

Evaluation can be done with `val` and `test` dataset. Make sure that your trained model is under the directory `Instance_Segmentation/outputs/snapshot`. 

For evaluation under `val`/`test`, run:
```
python3 segment_evaluate.py --dataset='test' --epochs='0100' --use_edgelist=0
python3 segment_evaluate.py --dataset='val' --epochs='0100' --use_edgelist=1
```

- You should set `--epochs` to the last four digits of the name of your trained model.
- Edgelist is used if setting `--use_edgelist=1`. **Note** that if you want to use edgelist as post-processing, make sure you have generated the edgelist labels following the instructions under `Instance_Segmentation/libs/edgelist_utils_matlab`. 

Our trained instance segmentation model can be download [here](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c?usp=sharing).


### Inference

You can obtain a instance segmentation output during inference. Inference can be done with `val` and `test` dataset.

For inference with the 2nd image in `val` dataset without edgelist, run:

```
python3 segment_inference.py --dataset='val' --image_id=2 --epochs='0100' --use_edgelist=0
```

- Inference under `test` dataset if setting `--dataset='test'`
- Try other image if setting `--image_id` to other number
- Set the `--epochs` to the last four digits of your trained model
- Edgelist is used if setting `--use_edgelist=1`. Also make sure the edgelist labels have been generated.

Also, you can try [our trained model](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c?usp=sharing).


### Visualization

You can visualize the ground-truth semantic results with the `.mat` data using `Instance_Segmentation/tools/instance_visualize.py`. **Note** that the data should be correctly arranged following the instructions under `data` directory.

For visualization with the 1st image in `train` dataset, run:

```
python3 instance_visualize.py --dataset='train' --image_id=1
```

- Visualization under `val`/`test` dataset if setting `--dataset='val'` or `--dataset='test'`
- Try other image if setting `--image_id` to other number



## Citation

Please cite the corresponding paper if you found our datasets or code useful:

```
@inproceedings{Zou18SketchyScene,
  author    = {Changqing Zou and
                Qian Yu and
                Ruofei Du and
                Haoran Mo and
                Yi-Zhe Song and
                Tao Xiang and
                Chengying Gao and
                Baoquan Chen and
                Hao Zhang},
  title     = {SketchyScene: Richly-Annotated Scene Sketches},
  booktitle = {ECCV},
  year      = {2018},
  publisher = {Springer International Publishing},
  pages		= {438--454},
  doi		= {10.1007/978-3-030-01267-0_26},
  url		= {https://github.com/SketchyScene/SketchyScene}
}
```

## Credits
- The ResNet-101 model pre-trained on ImageNet in TensorFlow is created by [chenxi116](https://github.com/chenxi116/TF-resnet)
- The code for the DeepLab model is authored by [Tensorflow authors](https://github.com/tensorflow/models/blob/master/research/resnet/resnet_model.py) and [chenxi116](https://github.com/chenxi116/TF-deeplab)
- The code for the Mask R-CNN model is authored by [matterport](https://github.com/matterport/Mask_RCNN)


## License

Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License with 996 ICU clause: [![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/en_US)

The above license is only granted to entities that act in concordance with local labor laws. In addition, the following requirements must be observed:

- The licensee must not, explicitly or implicitly, request or schedule their employees to work more than 45 hours in any single week.
- The licensee must not, explicitly or implicitly, request or schedule their employees to be at work consecutively for 10 hours.
