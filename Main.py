import models
import torch


darcknet53 = models.Darknet53()
fpn = models.FPN([256, 512, 1024], 256)
head = models.Head(256, 80, 3)

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# darcknet53.to(device)

inputs = torch.randn(1, 3, 608, 608) # Пример входных данных
outputs = head(fpn(darcknet53(inputs)))
print(*[f'{level} level: {output.shape}' for output, level in zip(outputs, ['Low', 'Mid', 'High'])])
