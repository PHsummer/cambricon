import numpy as np
# from nms import nms
from shapely.geometry import Polygon

class eval_cal(object):

    def __init__(self, iou_threshold):
        # self.pixel_threshold = float(args.pixel_threshold)
        self.iou_threshold = iou_threshold
        # self.quiet=args.quiet
        self.reset()

    def reset(self):
        self.img_num = 0
        self.pre = 0
        self.rec = 0
        self.f1_score = 0
        self.miou = 0

    def val(self):
        print(self.img_num)
        mpre = self.pre / self.img_num * 100
        mrec = self.rec / self.img_num * 100
        mf1_score = self.f1_score / self.img_num * 100
        miou = self.miou / self.img_num * 100
        return mpre, mrec, mf1_score, miou

    def sigmoid(self, x):
        """`y = 1 / (1 + exp(-x))`"""
        return 1 / (1 + np.exp(-x))

    def get_iou(self, g, p):
        g = Polygon(g)
        p = Polygon(p)
        if not g.is_valid or not p.is_valid:
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter/union

    def eval_one(self, quad_after_nms, gt_xy):
        num_gts = len(gt_xy)
        num_quads = len(quad_after_nms)
        if num_quads == 0:
            return 0, 0, 0

        quad_flag = np.zeros(num_quads)  # 记录quad是否被匹配
        gt_flag = np.zeros(num_gts)  # 记录gt是否被匹配

        iou_cal = []
        geo = quad_after_nms[0]

        if gt_flag[0] == 0:
            gt_geo = gt_xy[0]
        iou = self.get_iou(geo, gt_geo)
        iou_cal.append(iou)
        if iou >= self.iou_threshold:
            gt_flag[0] = 1  # 记录被匹配的gt框
            quad_flag[0] = 1  # 记录被匹配的quad框

        tp = np.sum(quad_flag)
        fp = num_quads - tp
        fn = num_gts - tp
        pre = tp / (tp + fp)  # 查准率（精确率）
        rec = tp / (tp + fn)  # 查全率（召回率）
        if pre + rec == 0:
            f1_score = 0
        else:
            f1_score = 2 * pre * rec / (pre + rec)
        miou = np.mean(iou_cal)
        # print(pre, '---', rec, '---', f1_score, '---', miou )
        return pre, rec, f1_score, miou

    def add(self, out, gt_xy_list):
        self.img_num += len(gt_xy_list)
        pre, rec, f1_score, miou = self.eval_one(out, gt_xy_list)
        self.pre += pre
        self.rec += rec
        self.f1_score += f1_score
        self.miou += miou
