# Self-adjusting Dice Loss

This is an unofficial PyTorch implementation of the 
[Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/abs/1911.02855) paper.

## Usage

Installation

```bash
pip install sadice
```

Example

```python
import torch
from sadice import SelfAdjDiceLoss

criterion = SelfAdjDiceLoss()
logits = torch.rand(128, 10, requires_grad=True)
targets = torch.randint(0, 10, size=(128, ))

loss = criterion(logits, targets)
loss.backward()
```