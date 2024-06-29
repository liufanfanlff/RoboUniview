"""Various loss implementations"""

import torch
from torch import nn
import torch.nn.functional as F


_l1_loss = nn.L1Loss(reduction="none")

def l1_loss(preds: torch.Tensor, trues: torch.Tensor, masks: torch.Tensor, norm_by_mask: bool = True):
    loss = _l1_loss(preds, trues)
    loss *= masks
    loss = loss.sum()
    if norm_by_mask and masks.sum() != 0:
        # replace this to torch.count_nonzero in pytorch 1.9
        loss /= (masks > 0).sum()
    return loss


class FocalLoss(nn.Module):
    def __init__(self, occlusion_mask_type=None, neg=False):
        super(FocalLoss, self).__init__()
        self.neg = neg
        self.occlusion_mask_type = occlusion_mask_type

    def forward(self, pred, gt, mask=None):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''
        if mask is not None:
            if self.occlusion_mask_type == 'all_ignore':
                gt[mask == 0] = -1
            elif self.occlusion_mask_type == 'all_neg':
                gt[mask == 0] = 0
            elif self.occlusion_mask_type == 'pos_ignore':
                gt[(mask == 0) * (gt == 1)] = -1
            elif self.occlusion_mask_type == 'other':
                pass
            else:
                pass
        pred=F.sigmoid(pred)

        non_ignore = gt != -1
        if gt.shape == pred.shape:
            pred = pred[non_ignore]
        else:
            pred = pred[non_ignore[:, None, :, :, :]]
        gt = gt[non_ignore].float()

        if len(pred) == 0:
            return 0

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)
        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if self.neg:
            neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        else:
            neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds
        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class Balanced_CE_loss(nn.Module):

    def __init__(self, pos_weight, occlusion_mask_type, reduction='mean'):
        super(Balanced_CE_loss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.occlusion_mask_type = occlusion_mask_type

    def forward(self, input, target, mask=None):
        if mask is not None:
            if self.occlusion_mask_type == 'all_ignore':
                target[mask == 0] = -1
            elif self.occlusion_mask_type == 'all_neg':
                target[mask == 0] = 0
            elif self.occlusion_mask_type == 'pos_ignore':
                target[(mask == 0) * (target == 1)] = -1
            else:
                raise NotImplementedError(f"{self.occlusion_mask_type} mask type not implement!")

        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg
        alpha_pos = num_neg / num_total * self.pos_weight
        alpha_neg = num_pos / num_total
        weights = torch.Tensor([alpha_neg, alpha_pos]).cuda()
        return F.cross_entropy(input, target, weights, ignore_index=-1, reduction=self.reduction)


class Balanced_BCE_loss(nn.Module):

    def __init__(self, pos_weight, reduction='mean'):
        super(Balanced_BCE_loss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, input_, target, mask=None):
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg
        alpha_pos = num_neg / (num_total * self.pos_weight + 1e-6)
        alpha_neg = num_pos / (num_total + 1e-6)
        pos_weight = alpha_pos / (alpha_neg + 1e-6)
        non_ignore = target != -1
        if target.shape == input_.shape:
            input_ = input_[non_ignore]
        else:
            input_ = input_[non_ignore[:, None, :, :, :]]
        if len(input_) == 0:
            return 0
        target = target[non_ignore].float()
        return F.binary_cross_entropy_with_logits(input_, target, pos_weight=pos_weight, reduction=self.reduction)


class CELossIgnore(nn.Module):
    def __init__(self, cls_weight, reduction="mean"):
        if isinstance(cls_weight, (list, tuple)):
            cls_weight = torch.tensor(cls_weight)
        assert isinstance(cls_weight, torch.Tensor), (
            "cls_weight in CELoss must be list, tuple or Tensor, "
            f"but got invalid type {cls_weight.__class__.__name__}"
        )
        assert reduction in ("mean", "sum")

        super(CELossIgnore, self).__init__()
        self.register_buffer("cls_weight", cls_weight, persistent=False)
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        gt = gt.long()
        if mask is not None:
            gt[mask == 0] = -1
        loss = F.cross_entropy(pred, gt, self.cls_weight, ignore_index=-1, reduction=self.reduction)

        return loss


class CELossIgnoreSem(nn.Module):
    def __init__(self, occlusion_mask_type, reduction="mean"):
        assert reduction in ("mean", "sum")
        super(CELossIgnoreSem, self).__init__()
        self.reduction = reduction
        self.occlusion_mask_type = occlusion_mask_type
        
    def forward(self, pred, target, mask=None):
        target = target.long()
        if mask is not None:
            if self.occlusion_mask_type == 'all_ignore':
                target[mask == 0] = -1
            elif self.occlusion_mask_type == 'all_neg':
                target[mask == 0] = 0
            elif self.occlusion_mask_type == 'pos_ignore':
                target[(mask == 0) * (target == 1)] = -1
            else:
                raise NotImplementedError(f"{self.occlusion_mask_type} mask type not implement!")
        loss = F.cross_entropy(pred, target, ignore_index=-1, reduction=self.reduction)
        return loss


class CELoss(nn.Module):
    def __init__(self, cls_weight, reduction="mean"):
        if isinstance(cls_weight, (list, tuple)):
            cls_weight = torch.tensor(cls_weight)
        assert isinstance(cls_weight, torch.Tensor), (
            "cls_weight in CELoss must be list, tuple or Tensor, "
            f"but got invalid type {cls_weight.__class__.__name__}"
        )
        assert reduction in ("mean", "sum")
        super(CELoss, self).__init__()
        self.register_buffer("cls_weight", cls_weight, persistent=False)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.cls_weight, reduction="none")
        # self.ce_loss = nn.CrossEntropyLoss(weight=[10., 50.], reduction="none")
        self.reduction = reduction

    def forward(self, preds, trues, masks=None):
        trues = trues.long()
        self.ce_loss.weight = self.cls_weight.to(preds.device)
        self.cls_weight = self.cls_weight.to(preds.device)
        loss = self.ce_loss(preds, trues)  # (N, X, Y, Z)
        loss = torch.sum(loss * masks)
        avg_weight = (
                torch.where(trues == 1, self.cls_weight[1], self.cls_weight[0]) * masks
        )  # average by weight
        loss = loss / avg_weight.sum()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def make_one_hot(self, labels, classes):
        # label size: torch.Size([2, 400, 400, 40])
        one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3], labels.size()[4]).zero_().to(labels.device)
        target = one_hot.scatter_(1, labels.data, 1)
        return target

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        target = torch.unsqueeze(target, dim=1)
        target = self.make_one_hot(target, 2)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        loss = 1 - num / den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def smooth_reg_loss(regr, gt_regr, mask=None, sigma=3, weight=None, reduction="mean"):
    if mask is not None:
        regr = regr * mask
        gt_regr = gt_regr * mask
    abs_diff = torch.abs(regr - gt_regr)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + (
            abs_diff - 0.5 / (sigma ** 2)
    ) * (1.0 - abs_diff_lt_1)
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))