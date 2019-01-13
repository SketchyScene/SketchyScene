import os
import scipy.io
import scipy.ndimage
import numpy as np
from PIL import Image


def label2d_array_nn_scaling(label2d, new_h, new_w):
    """
    implement nearest neighbor scaling for 2d array
    :param label2d: [H, W]
    :return: label_new: [new_h, new_w]
    """
    scale_h = new_h / label2d.shape[0]
    scale_w = new_w / label2d.shape[1]
    label_new = scipy.ndimage.zoom(label2d, zoom=[scale_h, scale_w], order=0)
    return label_new


def generate_edgelist(image_id, data_type, sketchyscene_base_dir, edgelist_base_dir):
    assert data_type in ['test', 'val']
    drawing_base_path = os.path.join(sketchyscene_base_dir, data_type, 'DRAWING_GT')
    edgelist_mat_path = os.path.join(edgelist_base_dir, data_type, 'edgelist_' + str(image_id) + '.mat')

    edge_lists = scipy.io.loadmat(edgelist_mat_path)['labelededgeim']
    edge_lists = np.array(edge_lists, dtype=np.float32)  # (750, 750), [0, nEdges]

    image_path = os.path.join(drawing_base_path, 'L0_sample' + str(image_id) + '.png')
    sketch = Image.open(image_path).convert('RGB')
    sketch_mask = np.array(sketch, dtype=np.uint8)[:, :, 0]

    sketch_edge_list = np.zeros(sketch_mask.shape, dtype=np.float32)

    for i in range(sketch_mask.shape[0]):
        for j in range(sketch_mask.shape[1]):
            if sketch_mask[i][j] != 0:
                continue

            # foreach black pixel, find the nearest edgelist label
            pixel_edge_label = []
            neighbor = 0
            while True:
                neighbor += 1

                for m in range(-neighbor, neighbor + 1):
                    for n in range(-neighbor, neighbor + 1):
                        pos_y = min(max(i + m, 0), sketch_mask.shape[0] - 1)
                        pos_x = min(max(j + n, 0), sketch_mask.shape[1] - 1)

                        if abs(pos_y - i) == neighbor or abs(pos_x - j) == neighbor:
                            if edge_lists[pos_y][pos_x] != 0:
                                pixel_edge_label.append(edge_lists[pos_y][pos_x])

                if len(pixel_edge_label) != 0:
                    break

            assert len(pixel_edge_label) != 0
            if len(pixel_edge_label) > 0:
                if j - 1 >= 0 and sketch_edge_list[i][j - 1] in pixel_edge_label:
                    sketch_edge_list[i][j] = sketch_edge_list[i][j - 1]
                elif i - 1 >= 0 and sketch_edge_list[i - 1][j] in pixel_edge_label:
                    sketch_edge_list[i][j] = sketch_edge_list[i - 1][j]
                else:
                    sketch_edge_list[i][j] = pixel_edge_label[0]
            else:
                sketch_edge_list[i][j] = pixel_edge_label[0]

    return sketch_edge_list


def refine_mask_with_edgelist(image_id, dataset_type, data_base_dir, edgelist_result_dir,
                              origin_mask, ori_boxes, pixels_percent_threshold=0.5):
    """
    very slow!!
    :param origin_mask: (768, 768, nRoIs)
    :param ori_boxes: (nRoIs, (y1, x1, y2, x2))
    :param pixels_percent_threshold: only with pixels_percent more than this threshold
             can be regarded as the same segment
    :return:
    """
    ## first generate edgelist
    edgelist_label = generate_edgelist(image_id, dataset_type, data_base_dir, edgelist_result_dir)  # (750, 750)
    edgelist_label = label2d_array_nn_scaling(edgelist_label,
                                              origin_mask.shape[0], origin_mask.shape[1])  # (768, 768)

    nEdgeList = int(np.max(edgelist_label))
    print('nEdgeList', nEdgeList)
    ori_mask = np.transpose(origin_mask, [2, 0, 1])  # (nRoIs, 768, 768)
    refined_mask = []
    for i in range(ori_mask.shape[0]):
        ## foreach ROI
        single_mask = ori_mask[i]
        single_mask_new = single_mask.copy()
        y1, x1, y2, x2 = ori_boxes[i]
        single_edgelist = edgelist_label.copy()
        box_bin_mask = np.zeros(single_edgelist.shape, dtype=np.uint8)
        box_bin_mask[y1: y2, x1: x2] = 1
        single_edgelist = single_edgelist * box_bin_mask

        for j in range(1, nEdgeList):
            if j not in single_edgelist:
                continue

            pos = single_edgelist != j
            single_mask_t = single_mask.copy()
            single_mask_t[pos] = 0
            label_count = np.bincount(single_mask_t.flatten())
            assert len(label_count) == 1 or len(label_count) == 2
            edge_pixels_count = np.sum(single_edgelist == j)
            if len(label_count) == 2 and label_count[1] > 0 \
                    and label_count[1] / edge_pixels_count >= pixels_percent_threshold:
                # print('refined here!')
                single_mask_new[single_edgelist == j] = 1

        single_mask_new = np.expand_dims(single_mask_new, axis=0)
        refined_mask.append(single_mask_new)

    refined_mask = np.concatenate(refined_mask, axis=0)
    refined_mask = np.transpose(refined_mask, [1, 2, 0])  # (768, 768, nRoIs)
    assert refined_mask.shape[2] == origin_mask.shape[2]

    return refined_mask
