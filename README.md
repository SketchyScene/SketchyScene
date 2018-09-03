# SketchyScene: Richly-Annotated Scene Sketches.

We contribute the first large-scale dataset of _scene sketches_, **SketchyScene**, with the goal of advancing research on sketch understanding at both the object and scene level. The dataset is created through a novel and carefully designed _crowdsourcing_ pipeline, enabling users to efficiently generate large quantities of realistic and diverse scene sketches. **SketchyScene** contains more than 29,000 scene-level sketches, 7,000+ pairs of scene templates and photos, and 11,000+ object sketches. All objects in the scene sketches have ground-truth semantic and instance masks. The dataset is also highly scalable and extensible, easily allowing augmenting and/or changing scene composition. We demonstrate the potential impact of **SketchyScene** by training new computational models for semantic segmentation of scene sketches and showing how the new dataset enables several applications including image retrieval, sketch colorization, editing, and captioning, etc. The dataset and code can be found at https://github.com/SketchyScene/SketchyScene.

Please cite the corresponding paper if you found our datasets or code useful:

> SketchyScene: Richly-Annotated Scene Sketches. Changqing Zou, Qian Yu, Ruofei Du, Haoran Mo, Yi-Zhe Song, Tao Xiang, Chengying Gao, Baoquan Chen, Hao Zhang. In Proceedings of European Conference on Computer Vision (ECCV), 2018.

## Dataset

[7265 (train 5617 + val 535 + test 1113) Downloading](https://drive.google.com/open?id=1m1fac2XIZVAGu_ByE6BtwxGSLytZspO-)

(Further data will come soon)

## USketch

**USketch** is a web-driven tool for crowdsourcing the **SketchyScene** dataset. It is open-sourced at https://github.com/ruofeidu/USketch. A live demo is located at http://duruofei.com/skenew/?task=1.

![USketch Interface](figures/USketch.jpg "Interface and work flow of USketch for crowdsourcing the dataset. See areas of function buttons (upper left), component display (lower left), and canvas (right). ")


## DuCrawler

In preparation for the reference data in **USketch**, we have developed a custom image crawler to acquire 7,500 cartoon scenes and 9,290 sketchy objects in 45 categories for academic use. Due to the length limitation, we explain the detailed crawling process and source code in the supplementary materials.

![Object instance frequency](figures/DuCrawler.jpg "Object instance frequency for each category.")

We selected 45 categories for our dataset, including objects and stuff classes. Specifically, we first considered several common scenes (e.g., garden, farm, dinning room, and park) and extracted 100 objects/stuff classes from them as raw candidates. Then we defined three super-classes, i.e. Weather, Object, and Field (Environment), and assigned the candidates into each super-class. Finally, we selected 45 from them by considering their combinations and commonness in real life. 

Instead of asking workers to draw each object, we provided them with plenty of object sketches (each object candidate is also refer to a ``component") as candidates. In order to have enough variations in the object appearance in terms of pose and appearance, we searched and downloaded around 1,500 components for each category. 

