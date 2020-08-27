import numpy as np
import glob

from util.data_util import load_img, load_pc, calibration_info


class CalibPcImg:
    def __init__(self):
        # pc_paths = glob.glob('./dataset/velodyne_points/*.bin')
        pc_paths = glob.glob('./Data-000002/lidar/*.bin')
        self.pc_paths = sorted(pc_paths)

        # img_paths = glob.glob('./dataset/image_02/*.png')
        img_paths = glob.glob('./Data-000002/image/*.jpeg')
        self.img_paths = sorted(img_paths)

        self.raw_pt = load_pc(self.pc_paths)
        self.raw_img = load_img(self.img_paths)
        self.trans_mat = calibration_info()

    def calib(self, frame):
        x3d = self.raw_pt[frame][:, 0]
        y3d = self.raw_pt[frame][:, 1]
        z3d = self.raw_pt[frame][:, 2]
        i3d = self.raw_pt[frame][:, 3]

        f_x = np.zeros(len(x3d), dtype=np.float32)
        f_y = np.zeros(len(y3d), dtype=np.float32)
        f_z = np.zeros(len(z3d), dtype=np.float32)
        f_i = np.zeros(len(i3d), dtype=np.float32)

        c_imgs = []

        iter = range(len(self.raw_pt[frame][:, 0]))

        img_width = self.raw_img[frame].shape[1]
        img_height = self.raw_img[frame].shape[0]

        arr = np.array
        dot = np.dot

        c_img = self.raw_img[frame]
        for i in iter:
            point = arr([[x3d[i]], [y3d[i]], [z3d[i]], [1]])
            point_t = dot(self.trans_mat, point)

            if point_t[2, 0] > 0:
                img_u = int(point_t[0, 0] / point_t[2, 0])
                img_v = int(point_t[1, 0] / point_t[2, 0])

                if 0 <= img_u < img_width and 0 <= img_v < img_height:
                    f_x[i] = x3d[i]
                    f_y[i] = y3d[i]
                    f_z[i] = z3d[i]
                    f_i[i] = i3d[i]

                    c_img[img_v, img_u] = [0, 0, 255]
                    c_imgs.append(c_img)

        f_pc = np.array([f_x.tolist(), f_y.tolist(), f_z.tolist(), f_i.tolist()], dtype=np.float32).transpose()

        return f_pc, c_imgs
