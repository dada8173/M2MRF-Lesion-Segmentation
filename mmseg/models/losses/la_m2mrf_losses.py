import torch
import torch.nn as nn

from ..builder import LOSSES
from .binary_loss import _make_one_hot
from .utils import weight_reduce_loss


def _prepare_binary_prediction(pred_raw, label_raw):
    pred = pred_raw.clone()
    label = label_raw.clone()
    num_classes = pred.shape[1]
    if pred.shape != label.shape:
        label = _make_one_hot(label, num_classes)
    pred = torch.sigmoid(pred)
    return pred, label.float(), num_classes


def _reduce_weighted_class_loss(class_losses,
                                weight=None,
                                class_weight=None,
                                reduction='mean',
                                avg_factor=None):
    loss = 0.
    for i, class_loss in enumerate(class_losses):
        if class_weight is not None:
            class_loss = class_loss * class_weight[i]
        loss += class_loss

    if class_weight is not None:
        loss = loss / torch.sum(class_weight)
    else:
        loss = loss / len(class_losses)
    return weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def weighted_binary_dice_loss(pred_raw,
                              label_raw,
                              weight=None,
                              class_weight=None,
                              reduction='mean',
                              avg_factor=None,
                              smooth=1e-5):
    pred, label, num_classes = _prepare_binary_prediction(pred_raw, label_raw)
    class_losses = []
    for i in range(num_classes):
        pred_i = pred[:, i].contiguous().view(pred.shape[0], -1)
        label_i = label[:, i].contiguous().view(label.shape[0], -1)
        numerator = 2 * torch.sum(pred_i * label_i, dim=1) + smooth
        denominator = torch.sum(pred_i, dim=1) + torch.sum(label_i, dim=1) + smooth
        class_losses.append(1. - numerator / denominator)
    return _reduce_weighted_class_loss(
        class_losses,
        weight=weight,
        class_weight=class_weight,
        reduction=reduction,
        avg_factor=avg_factor)


def focal_tversky_loss(pred_raw,
                       label_raw,
                       weight=None,
                       class_weight=None,
                       reduction='mean',
                       avg_factor=None,
                       alpha=0.3,
                       beta=0.7,
                       gamma=0.75,
                       smooth=1e-5):
    pred, label, num_classes = _prepare_binary_prediction(pred_raw, label_raw)
    class_losses = []
    for i in range(num_classes):
        pred_i = pred[:, i].contiguous().view(pred.shape[0], -1)
        label_i = label[:, i].contiguous().view(label.shape[0], -1)
        true_pos = torch.sum(pred_i * label_i, dim=1)
        false_neg = torch.sum(label_i * (1 - pred_i), dim=1)
        false_pos = torch.sum((1 - label_i) * pred_i, dim=1)
        tversky = (true_pos + smooth) / (
            true_pos + alpha * false_neg + beta * false_pos + smooth)
        class_losses.append(torch.pow(1. - tversky, gamma))
    return _reduce_weighted_class_loss(
        class_losses,
        weight=weight,
        class_weight=class_weight,
        reduction=reduction,
        avg_factor=avg_factor)


@LOSSES.register_module()
class WeightedBinaryDiceLoss(nn.Module):
    def __init__(self,
                 loss_type='dice',
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 smooth=1e-5,
                 **kwargs):
        super(WeightedBinaryDiceLoss, self).__init__()
        if loss_type != 'dice':
            raise ValueError('WeightedBinaryDiceLoss only supports loss_type="dice".')
        self.loss_type = loss_type
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.smooth = smooth

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            assert class_weight.shape[0] == cls_score.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(cls_score.shape[1], class_weight.shape[0])
        else:
            class_weight = None
        return self.loss_weight * weighted_binary_dice_loss(
            cls_score,
            label,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth)


@LOSSES.register_module()
class FocalTverskyLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 alpha=0.3,
                 beta=0.7,
                 gamma=0.75,
                 loss_weight=1.0,
                 smooth=1e-5,
                 **kwargs):
        super(FocalTverskyLoss, self).__init__()
        self.reduction = reduction
        self.class_weight = class_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.smooth = smooth

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            assert class_weight.shape[0] == cls_score.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(cls_score.shape[1], class_weight.shape[0])
        else:
            class_weight = None
        return self.loss_weight * focal_tversky_loss(
            cls_score,
            label,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            smooth=self.smooth)
