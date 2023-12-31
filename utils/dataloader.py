import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root, annotation_file):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = {}

        image_path = coco.loadImgs(img_id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target["boxes"] = [ann["bbox"] for ann in anns]
        target["labels"] = [ann["category_id"] for ann in anns]

        return self.__transform(image, target)

    @staticmethod
    def __transform(image, target, scale=608):
        scaled_boxes = []
        for box in target["boxes"]:
            x, y, w, h = box
            # Разделение координат на ширину и высоту исходного изображения
            x /= image.shape[1]
            y /= image.shape[0]
            w /= image.shape[1]
            h /= image.shape[0]
            # Умножение на новую ширину и высоту
            x *= scale
            y *= scale
            w *= scale
            h *= scale
            scaled_boxes.append([round(diget, 3) for diget in [x, y, w, h]])
        target["boxes"] = scaled_boxes
        image = cv2.resize(image, dsize=(scale, scale), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        return image, target

    def __len__(self):
        return len(self.ids)


def __collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack([torch.from_numpy(image.transpose((2, 0, 1))) for image in images])
    return images, targets


# Создание экземпляра DataLoader
def dataloader(data_root: str, annotation_file: str, batch_size: int = 32, shuffle: bool = True,
               fn: object = __collate_fn):
    """

    :param data_root: Путь к папке с изображениями
    :param annotation_file: Путь к файлу с аннотациями
    :param batch_size: Размер батча
    :param shuffle: Перемешивание
    :param fn: Предобработчик коллекции
    :return: Возвращает загрусчик данных для обучения и валидации
    """
    return DataLoader(CocoDataset(data_root, annotation_file), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=fn)
