import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        self.anchors = anchors

    def forward(self, output, target):
        mask, not_mask = target[..., 4:5], 1 - target[..., 4:5]

        pred_boxes = output[..., :4].sigmoid()
        target_boxes = target[..., :4]

        pred_conf = output[..., 4].sigmoid()
        target_conf = target[..., 4]

        pred_cls = output[..., 5:].sigmoid()
        target_cls = target[..., 5:]

        loss_box = self.mse_loss(mask * pred_boxes, mask * target_boxes)
        loss_conf_obj = self.bce_loss(mask * pred_conf, mask * target_conf)
        loss_conf_noobj = self.bce_loss(not_mask * pred_conf, not_mask * target_conf)
        loss_cls = self.bce_loss(mask * pred_cls, mask * target_cls)

        iou_anchors = self._calculate_iou(pred_boxes, target_boxes)
        mask_iou = iou_anchors > 0.5

        loss_xy = self.mse_loss(mask * mask_iou.float() * pred_boxes[..., :2],
                                mask * mask_iou.float() * target_boxes[..., :2])
        loss_wh = self.mse_loss(mask * mask_iou.float() * pred_boxes[..., 2:],
                                mask * mask_iou.float() * target_boxes[..., 2:])

        loss = (self.lambda_coord * (
                    loss_xy + loss_wh) + loss_conf_obj + self.lambda_noobj * loss_conf_noobj + loss_cls)

        return loss

    def _calculate_iou(self, boxes1, boxes2):
        b1_center_x = boxes1[..., 0]
        b1_center_y = boxes1[..., 1]
        b1_width = boxes1[..., 2]
        b1_height = boxes1[..., 3]

        b2_center_x = boxes2[..., 0]
        b2_center_y = boxes2[..., 1]
        b2_width = boxes2[..., 2]
        b2_height = boxes2[..., 3]

        b1_x1 = b1_center_x - b1_width / 2
        b1_y1 = b1_center_y - b1_height / 2
        b1_x2 = b1_center_x + b1_width / 2
        b1_y2 = b1_center_y + b1_height / 2

        b2_x1 = b2_center_x - b2_width / 2
        b2_y1 = b2_center_y - b2_height / 2
        b2_x2 = b2_center_x + b2_width / 2
        b2_y2 = b2_center_y + b2_height / 2

        intersect_x1 = torch.maximum(b1_x1, b2_x1)
        intersect_y1 = torch.maximum(b1_y1, b2_y1)
        intersect_x2 = torch.minimum(b1_x2, b2_x2)
        intersect_y2 = torch.minimum(b1_y2, b2_y2)

        intersect_area = torch.clamp(intersect_x2 - intersect_x1 + 1, min=0) * torch.clamp(
            intersect_y2 - intersect_y1 + 1, min=0)

        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height

        iou = intersect_area / (b1_area + b2_area - intersect_area + 1e-16)

        return iou