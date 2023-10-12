import torch

def nms_boxes(boxes, scores, threshold):
    # Получение координат углов прямоугольников
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Вычисление площи прямоугольников
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Сортировка прямоугольников по убыванию значений scores
    _, indices = scores.sort(descending=True)

    keep = []
    while indices.numel() > 0:
        # Получение индекса прямоугольника с наивысшим scores
        i = indices[0]

        # Добавление индекса этого прямоугольника в список keep
        keep.append(i.item())

        # Получение координат углов наивысшего прямоугольника
        x1_max = torch.max(x1[i], x1[indices[1:]])
        y1_max = torch.max(y1[i], y1[indices[1:]])
        x2_min = torch.min(x2[i], x2[indices[1:]])
        y2_min = torch.min(y2[i], y2[indices[1:]])

        # Вычисление ширины и высоты пересечения прямоугольников
        intersection_width = torch.clamp(x2_min - x1_max + 1, min=0)
        intersection_height = torch.clamp(y2_min - y1_max + 1, min=0)

        # Вычисление площи пересечения и пересечения / объединения
        intersection_area = intersection_width * intersection_height
        union_area = areas[i] + areas[indices[1:]] - intersection_area
        iou = intersection_area / union_area

        # Удаление прямоугольников с IoU выше заданного порогового значения
        indices = indices[1:][iou <= threshold]

    return keep
