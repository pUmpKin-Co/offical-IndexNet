import os
import sys
import torch
import logging
import functools
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()
def get_logger(name, output=None, distributed_rank=0, color=True, abbrev_name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def save_check_point(path, model, optimizer, epoch, epoch_loss, scheduler=None, best_score=None, save_backbone=True, sampler=None, dist=False):
    filename = "checkpoint" + "_" + str(epoch)
    checkpoint = {
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "loss": epoch_loss
    }
    if dist:
        checkpoint["model"] = model.module.state_dict()
    else:
        checkpoint["model"] = model.state_dict()

    if save_backbone:
        if dist is False:
            checkpoint["backbone"] = model.target_encoder.extract_backbone_checkpoint()
        else:
            checkpoint["backbone"] = model.module.target_encoder.extract_backbone_checkpoint()
    if scheduler is not None:
        checkpoint["lr_schedule"] = scheduler.state_dict()
    if best_score is not None:
        checkpoint["best_score"] = best_score
    if sampler is not None:
        checkpoint["amp"] = sampler.state_dict()


    path_name = os.path.join(path, filename)
    torch.save(checkpoint, path_name)


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")