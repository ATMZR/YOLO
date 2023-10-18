import models
import utils

if __name__ == '__main__':
    darcknet53 = models.Darknet53()
    fpn = models.FPN([256, 512, 1024], 256)
    head = models.Head(256, 80, 3)

    for images, targets in utils.dataloader:
        pass
