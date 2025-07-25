import math

def cosine_annealing_with_warmup(epoch, lr, total_epochs=100, warmup_epochs=5, min_lr=1e-6, max_lr = 1e-3):
    """
    Cosine annealing with warmup for segmentation training
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))