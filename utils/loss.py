import torch.nn as nn
class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes, anchors, lambda_coord, lambda_noobj):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, output, target):
        # Расчет loss для ограничивающих рамок
        loss_coord = self.__calculate_coord_loss(output, target, self.lambda_coord)

        # Расчет loss для уверенности модели
        loss_conf = self.__calculate_confidence_loss(output, target, self.lambda_noobj)

        # Расчет loss для классификации объектов
        loss_class = self.__calculate_class_loss(output, target)

        # Общий loss
        total_loss = loss_coord + loss_conf + loss_class

        return total_loss

    @staticmethod
    def __calculate_coord_loss(output, target, lambda_coord):
        mask = target[:, 4:5]
        not_mask = 1 - mask

        pred_boxes = output[:, :4].sigmoid()
        target_boxes = target[:, :4]

        loss_x = nn.MSELoss()(mask * pred_boxes[:, 0], mask * target_boxes[:, 0])
        loss_y = nn.MSELoss()(mask * pred_boxes[:, 1], mask * target_boxes[:, 1])
        loss_w = nn.MSELoss()(mask * pred_boxes[:, 2], mask * target_boxes[:, 2])
        loss_h = nn.MSELoss()(mask * pred_boxes[:, 3], mask * target_boxes[:, 3])

        loss_coord = lambda_coord * (loss_x + loss_y + loss_w + loss_h)

        return loss_coord

    @staticmethod
    def __calculate_confidence_loss(output, target, lambda_noobj):
        mask = target[:, 4:5]
        not_mask = 1 - mask

        pred_conf = output[:, 4].sigmoid()
        target_conf = target[:, 4]

        loss_obj = nn.BCELoss()(mask * pred_conf, mask * target_conf)
        loss_noobj = nn.BCELoss()(not_mask * pred_conf, not_mask * target_conf)

        loss_conf = loss_obj + lambda_noobj * loss_noobj

        return loss_conf

    @staticmethod
    def __calculate_class_loss(output, target):
        mask = target[:, 4:5]
        not_mask = 1 - mask

        pred_cls = output[:, 5:].sigmoid()
        target_cls = target[:, 5:]

        loss_cls = nn.BCELoss()(mask * pred_cls, mask * target_cls)

        return loss_cls