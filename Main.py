import models
import utils

if __name__ == '__main__':
    darcknet53 = models.Darknet53()
    fpn = models.FPN([256, 512, 1024], 256)
    head = models.Head(256, 80, 3)

    # Путь к корневой папке датасета COCO
    data_root = "..."
    # Путь к файлу с аннотациями
    annotation_file = "..."

    for images, targets in utils.dataloader(data_root, annotation_file, batch_size=128):
        pass
