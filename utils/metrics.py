import numpy as np


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetric(_StreamMetrics):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        return hist

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k not in ["Class IoU", "F1_score", "Class Acc", "Hist"]:
                string += "%s: %f\n" % (k, v)

        return string

    def get_results(self):
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / (hist.sum() + 1e-8)  # OA

        acc_cls = np.diag(hist) / (hist.sum(axis=0) + 1e-8)  # class OA

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-8 )
        mean_iu = np.nanmean(iu)  # mIoU

        freq = hist.sum(axis=1) / (hist.sum() + 1e-8)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        recall = np.diag(hist) / (hist.sum(axis=1) + 1e-8)

        F1_score = (2 * recall * acc_cls) / (acc_cls + recall + 1e-8)

        # kappa
        pe_rows = np.sum(hist, axis=0)
        pe_cols = np.sum(hist, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / (np.float32(sum_total ** 2) + 1e-8)
        po = np.trace(hist) / (np.float32(sum_total) + 1e-8)
        kappa = (po - pe) / (1 - pe)

        return {
            "Hist": hist,
            "Overall Acc": acc,
            "Mean Acc": np.nanmean(acc_cls),
            "Class Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "F1_score": F1_score,
            "kappa": kappa
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count