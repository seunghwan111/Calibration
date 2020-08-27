import numpy as np
import cv2


def load_img(img_paths):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        imgs.append(img)

    return imgs


def load_pc(pt_paths):
    pts = []
    for pt_path in pt_paths:
        pt = np.fromfile(pt_path, dtype=np.float32).reshape(-1, 4)
        pts.append(pt)

    return pts


def calibration_info():
    '''
    p_mat = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
                      [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
                      [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]],
                     dtype=np.float64)

    r_mat = np.array([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0],
                      [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0],
                      [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0],
                      [0, 0, 0, 1]],
                     dtype=np.float64)
                     
    t_mat = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                      [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                      [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                      [0, 0, 0, 1]],
                     dtype=np.float64)
    '''
    p_mat = np.array([7.215377e+02, 0, 6.095593e+02, 4.485728e+01, 0, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0, 0, 1, 2.745884e-03], dtype=np.float64)

    r_mat = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                      -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02], dtype=np.float64)

    r_mat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    t_mat = np.array([0, 0, -10.0], dtype=np.float64)

    p_mat = np.reshape(p_mat, (3, 4))
    p_mat = p_mat[:3, :3]
    r_mat = np.reshape(r_mat, (3, 3))
    t_mat = np.reshape(t_mat, (3, 1))

    rt_mat = np.concatenate((r_mat, t_mat), axis=1)

    # trans_mat = np.dot(p_mat, np.dot(r_mat, t_mat))
    trans_mat = np.dot(p_mat, rt_mat)

    return trans_mat
