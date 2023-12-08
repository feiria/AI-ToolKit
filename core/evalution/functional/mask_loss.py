import torch


def get_mask(self, probs, threshold):
    """
         将sigmoid低于阈值的idx mask掉
         eg:threshold=0.8  需要将prob>0.8 和prob<0.2 的保留
         probs=[0.81,0.7，0.4，0.19,0.3,0.7]
         的得到mask=[1,0,0,1,0,0]
    """
    probs = torch.reshape(probs, [-1])
    mask1 = torch.less(probs, 1 - threshold)
    mask2 = torch.greater(probs, threshold)
    return torch.logical_or(mask1, mask2)


def mask_loss(self, loss, probs):
    mask = self.get_mask(probs, self.mask_loss_thres)
    mask = mask.type(loss.dtype)
    recall = torch.mean(mask)
    loss = loss * mask
    '''
        扔掉了一部分数据的loss,所以需要将剩下的loss放大
        防止模型将结果全部预测到这个区间
    '''
    loss = divide_no_nan(loss, recall)
    return loss


def divide_no_nan(x, y):
    z = torch.div(x, y)
    z_nan = torch.isnan(z)
    z_inf = torch.isinf(z)
    z_true = torch.logical_or(z_nan, z_inf)
    z = torch.where(z_true, torch.zeros_like(z), z)
    return z
