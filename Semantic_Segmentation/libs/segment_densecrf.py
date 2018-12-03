import numpy as np
import os
import pydensecrf.densecrf as dcrf


def seg_densecrf(prob_arr_, im_arr_, nlabels):
    """
    :param prob_arr_: shape = [nlabels, H, W], contains prob
    :param im_arr_:  shape = [H, W, 3], dtype == np.uint8
    :param nlabels:
    :return label_map: shape=[H, W], contains [0 ~ nlabels-1]
    """
    H = im_arr_.shape[0]
    W = im_arr_.shape[1]

    prob_arr = np.array(prob_arr_, np.float32)
    prob_min = np.amin(prob_arr)
    # print("prob_min", prob_min)
    prob_arr -= prob_min  # [0, ?]
    prob_arr += 0.0001    # [0.0001, ?]

    d = dcrf.DenseCRF2D(W, H, nlabels)

    U = np.array(-np.log(prob_arr), dtype=np.float32)  #shape = (nlabels, W, H), dtype('float32')
    U = U.reshape((nlabels, -1))  # Needs to be flat.
    U = U.copy(order='C')
    # print(U.flags)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=7, srgb=3, rgbim=im_arr_, compat=10)

    Q = d.inference(5)
    label_map = np.argmax(Q, axis=0).reshape((H, W))  # shape=[H, W], contains [0 ~ nlabels-1]
    return label_map
