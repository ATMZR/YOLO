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
        target["boxes"] = [ann["bbox"] for ann in anns]
        target["labels"] = [ann["category_id"] for ann in anns]
        image_path = coco.loadImgs(img_id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(608, 608), interpolation=cv2.INTER_AREA)
        image = image / 255.0

        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack([torch.from_numpy(image.transpose((2, 0, 1))) for image in images])
    return images, targets


# Путь к корневой папке датасета COCO
data_root = "..."

# Путь к файлу с аннотациями
annotation_file = "..."

# Создание экземпляра датасета COCO
dataset = CocoDataset(data_root, annotation_file)

# Создание экземпляра DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)