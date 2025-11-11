import torch
import time
from attack_lib.pgd.pgd import attack_PGD
from data_process import normalization
from pruner import pruning


# Ours
def attack_ours(model, images, labels, conf, target=False):
    start_time = time.time()
    init_noises, _ = attack_PGD(
        model, images, labels, conf, target, iterations=80)
    noises, adv_images, _, _, change = pruning(
        model, init_noises, images, labels, conf, target=target)
    end_time = time.time()
    return noises, adv_images, end_time - start_time


def attack_batch(model, images, labels, conf, attacker, target=False):
    if attacker == "ours":
        return attack_ours(model, images, labels, conf, target)


def filter_correct(model, images, labels, conf, batch_size=None):
    batch_size = images.shape[0] if batch_size is None else min(
        batch_size, images.shape[0])
    epoch = images.shape[0] // batch_size
    for i in range(epoch):
        if i == epoch - 1:
            images_b = images[i*batch_size:]
            labels_b = labels[i*batch_size:]
        else:
            images_b = images[i*batch_size: (i+1)*batch_size]
            labels_b = labels[i*batch_size: (i+1)*batch_size]
        outputs = model(normalization(
            images_b.to(conf.device), conf.normalized_ops))
        _, preds = torch.max(outputs, 1)
        idx_b = (preds == labels_b.to(conf.device).data)
        idx = idx_b if i == 0 else torch.cat((idx, idx_b), 0)
    return images[idx], labels[idx], idx


def filter_adv(model, images, labels, conf, batch_size=None):
    batch_size = images.shape[0] if batch_size is None else min(
        batch_size, images.shape[0])
    epoch = images.shape[0] // batch_size
    for i in range(epoch):
        if i == epoch - 1:
            images_b = images[i*batch_size:]
            labels_b = labels[i*batch_size:]
        else:
            images_b = images[i*batch_size: (i+1)*batch_size]
            labels_b = labels[i*batch_size: (i+1)*batch_size]
        outputs = model(normalization(
            images_b.to(conf.device), conf.normalized_ops))
        _, preds = torch.max(outputs, 1)
        idx_b = (preds != labels_b.to(conf.device).data)
        idx = idx_b if i == 0 else torch.cat((idx, idx_b), 0)
    return images[idx], labels[idx], idx
