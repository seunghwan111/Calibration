from util.calib_pc_img import CalibPcImg
from util.viewer import Viewer


if __name__ == '__main__':
    print('Start!\n')

    calib_pt_img = CalibPcImg()
    viewer = Viewer()

    for frame in range(len(calib_pt_img.pc_paths)):
        pc, img = calib_pt_img.calib(frame)
        print(pc.shape)
        viewer.xyz_viewer(pc)
        viewer.img_viewer(img[frame])

        print('Frame [%d]' % (frame + 1))

    viewer.viewer_exit()

    print('\nDone.')
