import numpy as np
import torch

from sadice import SelfAdjDiceLoss


LOGITS_NUMPY = np.random.rand(128, 10)
TARGETS_NUMPY = np.random.randint(0, 10, size=(128, ))
LOGITS_TORCH = torch.from_numpy(LOGITS_NUMPY)
TARGETS_TORCH = torch.from_numpy(TARGETS_NUMPY)
ALPHA = 1.0
GAMMA = 1.0


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_loss(logits: np.ndarray, targets: np.ndarray, alpha: float = 1.0, gamma: float = 1.0) -> float:

    loss = 0.0
    for curr_logits, curr_target in zip(logits, targets):
        curr_probs = softmax(curr_logits)

        curr_prob = curr_probs[int(curr_target)]
        prob_with_factor = ((1 - curr_prob) ** alpha) * curr_prob
        curr_loss = 1 - (2 * prob_with_factor + gamma) / (prob_with_factor + 1 + gamma)
        loss += curr_loss

    return loss / logits.shape[0]


def test_numpy_vs_torch():
    criterion = SelfAdjDiceLoss(alpha=ALPHA, gamma=GAMMA)
    loss_torch = criterion(LOGITS_TORCH, TARGETS_TORCH).item()
    loss_numpy = calculate_loss(LOGITS_NUMPY, TARGETS_NUMPY, ALPHA, GAMMA)
    assert np.allclose(loss_numpy, loss_torch)
