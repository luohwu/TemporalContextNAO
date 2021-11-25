import torch
from torch import nn
import math

class CIOU_LOSS(nn.Module):
    def __init__(self):
        super(CIOU_LOSS, self).__init__()

    def forward(self, bboxes1, bboxes2):

        assert bboxes1.shape[0]==bboxes2.shape[0],f' # of predicts != # of targets'
        w1 = (bboxes1[:, 2] - bboxes1[:, 0])
        h1 = (bboxes1[:, 3] - bboxes1[:, 1])
        w2 = (bboxes2[:, 2] - bboxes2[:, 0])
        h2 = (bboxes2[:, 3] - bboxes2[:, 1])
        area1 = w1 * h1
        area2 = w2 * h2
        center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2
        center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2
        center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2
        center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2

        inter_l = torch.max(bboxes1[:, 0], bboxes2[:, 0])
        inter_r = torch.min(bboxes1[:, 2], bboxes2[:, 2])
        inter_t = torch.max(bboxes1[:, 1], bboxes2[:, 1])
        inter_b = torch.min(bboxes1[:, 3], bboxes2[:, 3])
        inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)
        # print(inter_area)

        c_l = torch.min(bboxes1[:, 0], bboxes2[:, 0])
        c_r = torch.max(bboxes1[:, 2], bboxes2[:, 2])
        c_t = torch.min(bboxes1[:, 1], bboxes2[:, 1])
        c_b = torch.max(bboxes1[:, 3], bboxes2[:, 3])

        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

        union = area1 + area2 - inter_area
        u = (inter_diag) / c_diag
        iou = inter_area / union
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        with torch.no_grad():
            S = (iou > 0.5).float()
            alpha = S * v / (1 - iou + v)
        cious = iou - u - alpha * v
        cious = torch.clamp(cious, min=-1.0, max=1.0)
        # return cious
        return torch.sum(1 - cious),iou


if __name__=='__main__':
    ciou_loss = CIOU_LOSS()

    bboxes1 = torch.tensor([[1, 2, 3, 4], [15, 15, 165, 165]])
    bboxes2 = torch.tensor([[2, 3, 4, 5], [100, 100, 200, 200]])
    # bboxes1 = torch.randint(100, (8, 4))
    # bboxes2 = torch.randint(100, (8, 4))

    loss = ciou_loss(bboxes1, bboxes2)
    print(loss)
