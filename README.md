# Self-adjusting Dice Loss

This is an unofficial PyTorch implementation of the 
[Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/abs/1911.02855) paper.

## Usage

Installation

```bash
pip install sadice
```

### Text classification example

```python
import torch
from sadice import SelfAdjDiceLoss

criterion = SelfAdjDiceLoss()
# (batch_size, num_classes)
logits = torch.rand(128, 10, requires_grad=True)
targets = torch.randint(0, 10, size=(128, ))

loss = criterion(logits, targets)
loss.backward()
```

### NER example

```python
import torch
from sadice import SelfAdjDiceLoss

criterion = SelfAdjDiceLoss(reduction="none")
# (batch_size, num_tokens, num_classes)
logits = torch.rand(128, 40, 10, requires_grad=True)
targets = torch.randint(0, 10, size=(128, 40))

loss = criterion(logits.view(-1, 10), targets.view(-1))
loss = loss.reshape(-1, 40).mean(-1).mean()
loss.backward()
```