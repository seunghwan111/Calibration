from pcl import PointCloud_PointXYZI
from pcl.pcl_visualization import CloudViewing

import cv2


class Viewer:
    def __init__(self):
        self.pc_viewer = CloudViewing()

    def xyz_viewer(self, point_cloud):
        pc = PointCloud_PointXYZI(point_cloud[:, :4])
        self.pc_viewer.ShowGrayCloud(pc)

    @staticmethod
    def img_viewer(image):
        cv2.imshow('img', image)
        cv2.waitKey(0)

    def viewer_exit(self):
        v = True

        while v:
            v = not (self.pc_viewer.WasStopped())
            cv2.destroyAllWindows()
