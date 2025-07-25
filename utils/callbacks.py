import math

def cosine_annealing_with_warmup(epoch, lr, total_epochs=100, warmup_epochs=5, min_lr=1e-6):
    """
    Cosine annealing with warmup for segmentation training
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))