import torch
import matplotlib.pyplot as plt
import math


def accuracy(output, target):
    compar = torch.argmax(output, 1) == target
    compar = compar.type(torch.FloatTensor)
    acc = torch.mean(compar)
    return acc


# def contrastive_loss(h1,h2,labels):
#     H = torch.cat([h1,h2],dim=0)
#     device = torch.device('cuda')
#     labels = labels.view(-1,1)
#     batch_size = h1.shape[0]

#     # tile mask
#     mask = torch.eq(labels, labels.T).float().to(device)
#     mask = mask.repeat(2, 2)
#     # mask-out self-contrast cases
#     logits_mask = torch.scatter(torch.ones_like(mask),1,
#                                 torch.arange(batch_size * 2).view(-1, 1).to(device),0)

#     anchor_dot_contrast = torch.div(torch.matmul(H, H.T),0.1)

#     mask = mask * logits_mask

#     # compute log_prob
#     exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
#     log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))

#     # compute mean of log-likelihood over positive
#     mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#     # loss
#     loss = - mean_log_prob_pos
#     loss = loss.view(2, batch_size).mean()

#     return loss


def contrastive_loss(features, labels):
    device = torch.device('cuda')
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        0.1)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -  mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if True:
        eta_min = lr * (0.1 ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epoch)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Prepocessing:
    "change gamma for image with low luminance"

    def __init__(self):
        pass

    def __call__(self, x):
        low_pixels = torch.mean((torch.mean(x, axis=0) < 0.2).type(torch.FloatTensor))
        if low_pixels.item() >= 0.5:
            I = torch.pow(x, 0.5)
            return I
        return x

