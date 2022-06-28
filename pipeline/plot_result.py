from PIL import Image
import numpy as np

class FS_color():
    def __init__(self):
        self.color={}
        from Working_Environment.environment_variables import FREESURFER_HOME
        text_file = open(FREESURFER_HOME + '/FreeSurferColorLUT.txt', 'r')
        self.color[-1] = [25, 250, 250]
        for text in text_file.readlines():
            try:
                slipts = text.split()
                if (int(slipts[-1] == '0')):
                    self.color[int(text.split(' ')[0])] = [int(slipts[-4]),
                                                           int(slipts[-3]),
                                                           int(slipts[-2])]
            except:
                continue
        self.color[99] = [241, 214, 145]
        self.color[100] = [111, 185, 210]
        self.color[101] = [128, 174, 128]

class Plot_result():
    def __init__(self, mri, mri_warp, pred1, pred2, gt, affine_matrix, is_synthesized=True):
        self.color = FS_color()
        self.mri = mri
        self.mri_warp = mri_warp
        self.pred1 = pred1
        self.pred1[np.where(gt==-1)] = -1
        self.pred2 = pred2
        self.pred2[np.where(gt==-1)] = -1
        gt[np.where(gt>=2000)] = 42
        gt[np.where(gt>=1000)] = 3
        self.gt = gt
        self.affine_matrix = affine_matrix
        self.rotation = affine_matrix[:3,:3]
        self.is_synthesized = is_synthesized
        self.transform_volumes()

    def __init__(self, mri, mri_warp, pred1, pred2, pred3, gt, affine_matrix, is_synthesized=True):
        self.color = FS_color()
        self.mri = mri
        self.mri_warp = mri_warp
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        gt[np.where(gt>=2000)] = 42
        gt[np.where(gt>=1000)] = 3
        self.gt = gt
        self.affine_matrix = affine_matrix
        self.rotation = affine_matrix[:3,:3]
        self.is_synthesized = is_synthesized
        self.transform_volumes()

    def transform_volumes(self):
        order = [1, 2, 3]
        order = self.rotation.dot(order)
        order_real = np.around(np.abs(order)).astype(np.int16)
        self.mri = np.transpose(self.mri, order_real-1)
        self.mri_warp = np.transpose(self.mri_warp, order_real - 1)
        self.pred1 = np.transpose(self.pred1, order_real-1)
        self.pred2 = np.transpose(self.pred2, order_real-1)
        self.pred3 = np.transpose(self.pred3, order_real - 1)
        self.gt = np.transpose(self.gt, order_real-1)
        rot = np.sum(self.rotation, axis=0)
        for i in range(3):
            if order[i] > 0:
                self.mri = np.flip(self.mri, i)
                self.mri_warp = np.flip(self.mri_warp, i)
                self.pred1 = np.flip(self.pred1, i)
                self.pred2 = np.flip(self.pred2, i)
                self.pred3 = np.flip(self.pred3, i)
                self.gt = np.flip(self.gt, i)
        self.get_tumor_center()

    def get_tumor_center(self):
        if self.is_synthesized:
            tumor_voxel_xyz = np.where(self.gt==-1)
        else:
            tumor_voxel_xyz = np.where(self.gt>0)
        stem_voxel_xyz = np.where(self.gt == 16)
        self.tumor_center = np.mean(tumor_voxel_xyz, axis=1).astype(np.uint8)

    def plot_mri(self, mri):
        slice0 = mri[self.tumor_center[0], :, :].T
        slice1 = mri[:, self.tumor_center[1], :].T
        slice2 = mri[:, :, self.tumor_center[2]].T
        slice4 = np.zeros(shape=[slice2.shape[0], slice0.shape[1]])
        out = np.concatenate([np.concatenate([slice2, slice4], axis=1), np.concatenate([slice1, slice0], axis=1)], axis=0)
        return np.tile(np.expand_dims(out, axis=2), [1, 1, 3])

    def plot_mri_brats(self, mri):
        slice0 = mri[self.tumor_center[0], :, :].T
        slice1 = mri[:, self.tumor_center[1], :].T
        slice2 = mri[:, :, self.tumor_center[2]].T
        out = np.concatenate([slice2, np.concatenate([slice1[::2,::2], slice0[::2,::2]], axis=1)], axis=0)
        return np.tile(np.expand_dims(out, axis=2), [1, 1, 3])

    def seg_to_color(self, seg):
        color_img = np.zeros(shape=[seg.shape[0], seg.shape[1], 3], dtype=np.uint8)
        labels = list(np.unique(seg))
        labels.remove(0)
        for label in labels:
            if not int(label) in self.color.color:
                labels.remove(label)
        for label in labels:
            if not int(label) in self.color.color:
                labels.remove(label)
        if 632.0 in labels:
            labels.remove(632.0)
        if 631.0 in labels:
            labels.remove(631.0)
        for label in labels:
            xy = np.where(seg==label)
            color_img[xy[0], xy[1], :] = self.color.color[label]
        return color_img

    def plot_seg(self, seg):
        slice0 = np.transpose(self.seg_to_color(seg[self.tumor_center[0], :, :]), [1, 0, 2])
        slice1 = np.transpose(self.seg_to_color(seg[:, self.tumor_center[1], :]), [1, 0, 2])
        slice2 = np.transpose(self.seg_to_color(seg[:, :, self.tumor_center[2]]), [1, 0, 2])
        slice4 = np.zeros(shape=[slice2.shape[0], slice0.shape[1], 3], dtype=np.uint8)
        seg_slices = np.concatenate([np.concatenate([slice2, slice4], axis=1), np.concatenate([slice1, slice0], axis=1)],axis=0)
        alpha = np.zeros(shape=[seg_slices.shape[0], seg_slices.shape[1]], dtype=np.float32)
        alpha[np.where(np.sum(seg_slices, axis=-1)!=0)] = 0.4
        return seg_slices, np.tile(np.expand_dims(alpha, axis=2), [1, 1, 3])

    def plot_seg_brats(self, seg):
        slice0 = np.transpose(self.seg_to_color(seg[self.tumor_center[0], :, :]), [1, 0, 2])
        slice1 = np.transpose(self.seg_to_color(seg[:, self.tumor_center[1], :]), [1, 0, 2])
        slice2 = np.transpose(self.seg_to_color(seg[:, :, self.tumor_center[2]]), [1, 0, 2])
        slice4 = np.zeros(shape=[slice2.shape[0], slice0.shape[1], 3], dtype=np.uint8)
        seg_slices = np.concatenate([slice2, np.concatenate([slice1[::2,::2, :], slice0[::2,::2, :]], axis=1)], axis=0)
        alpha = np.zeros(shape=[seg_slices.shape[0], seg_slices.shape[1]], dtype=np.float32)
        alpha[np.where(np.sum(seg_slices, axis=-1)!=0)] = 0.4
        return seg_slices, np.tile(np.expand_dims(alpha, axis=2), [1, 1, 3])

    def plot(self):
        img1 = self.plot_mri(self.mri)
        img1 = 256 * img1 / np.max(img1)
        img11 = self.plot_mri(self.mri_warp)
        img11 = 256 * img11 / np.max(img11)
        seg2, alpha2 = self.plot_seg(self.pred1)
        seg3, alpha3 = self.plot_seg(self.pred2)
        img_gt, alpha_gt = self.plot_seg(self.gt)
        img2 = (1 - alpha2) * img1 + alpha2 * seg2
        img3 = (1 - alpha3) * img1 + alpha3 * seg3
        img4 = (1 - alpha_gt) * img1 + alpha_gt * img_gt
        return np.concatenate([img1, img11, img2, img3, img4], axis=1).astype(np.uint8)

    def plot_brats(self):
        img1 = self.plot_mri(self.mri)
        img1 = 256 * img1 / np.max(img1)
        seg2, alpha2 = self.plot_seg(self.pred1)
        seg3, alpha3 = self.plot_seg(self.pred2)
        seg4, alpha4 = self.plot_seg(self.pred3)
        img2 = (1 - alpha2) * img1 + alpha2 * seg2
        img3 = (1 - alpha3) * img1 + alpha3 * seg3
        img4 = (1 - alpha4) * img1 + alpha4 * seg4
        return np.concatenate([img1, img2, img3, img4], axis=1).astype(np.uint8)


    def produce_img(self, output_file, metric1, metric2, metric1_nocortex, metric2_nocortex):
        import os, subprocess
        if os.path.isfile(output_file):
            subprocess.run(['rm', output_file])
        img = Image.fromarray(self.plot())

        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        img_names = ['org T1', 'warped T1', 'seg no warp', 'seg warp', 'ground truth']
        single_img_width = int(img.width / len(img_names))
        for i in range(len(img_names)):
            draw.text((i * single_img_width + 10, 10), img_names[i], (255, 255, 255))

        def plot_metrics(metrics, metrics2, org_xy, title=None):
            text_space = 90
            if title == None:
                draw.text((org_xy[0], org_xy[1]+15), "all", (255, 255, 255))
                draw.text((org_xy[0], org_xy[1]+30), "sub_cortical", (255, 255, 255))
                draw.text((org_xy[0]+text_space, org_xy[1]), 'overall', (255, 255, 255))
                draw.text((org_xy[0]+2*text_space, org_xy[1]), 'tumor side', (255, 255, 255))
                draw.text((org_xy[0]+3*text_space, org_xy[1]), 'non_tumor side', (255, 255, 255))
            else:
                draw.text((org_xy[0], org_xy[1]+15), title, (255, 255, 255))
                draw.text((org_xy[0], org_xy[1]+30), title, (255, 255, 255))
                draw.text((org_xy[0], org_xy[1] + 45), "(sub_cortical)", (255, 255, 255))
            draw.text((org_xy[0]+text_space, org_xy[1]+15), str(metrics[0])[:5], (255, 255, 255))
            draw.text((org_xy[0]+2*text_space, org_xy[1]+15), str(metrics[1])[:5], (255, 255, 255))
            draw.text((org_xy[0]+3*text_space, org_xy[1]+15), str(metrics[2])[:5], (255, 255, 255))
            draw.text((org_xy[0]+text_space, org_xy[1]+30), str(metrics2[0])[:5], (255, 255, 255))
            draw.text((org_xy[0]+2*text_space, org_xy[1]+30), str(metrics2[1])[:5], (255, 255, 255))
            draw.text((org_xy[0]+3*text_space, org_xy[1]+30), str(metrics2[2])[:5], (255, 255, 255))

        plot_metrics(metric1, metric1_nocortex, (2 * single_img_width + 10, img.height - 40))
        plot_metrics(metric2, metric2_nocortex, (3 * single_img_width + 10, img.height - 40))
        plot_metrics(100*np.asarray(metric2-metric1), 100*np.asarray(metric2_nocortex-metric1_nocortex), (4 * single_img_width + 10, img.height - 40), title='improvement')

        img.save(output_file)
        #img.show()
        return output_file

    def produce_img_brats(self, output_file):
        import os, subprocess
        if os.path.isfile(output_file):
            subprocess.run(['rm', output_file])
        img = Image.fromarray(self.plot_brats())
        img.save(output_file)
        #img.show()
        return output_file