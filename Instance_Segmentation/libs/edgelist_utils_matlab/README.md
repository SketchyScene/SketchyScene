# Edgelist utils in matlab

Before using edgelist for post-processing in Instance Segmentation, the edgelist labels should be generated firstly.

1. Run `edgelist_main.m` to generate the edgelist labels for all the scene sketch.

 - You need to specify the `dataset_type`, i.e. 'test' or 'val', at the head of this file.

 - All the generated edgelist labels will be placed under `SketchyScene/Instance_Segmentation/outputs/edgelist/`

1. You can run `edgelist_visualize.m` to visualize the edgelist labels for a scene sketch.

 - You need to specify the `dataset_type` and the `image_id` at the head of this file.