"""
Refs:
    https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/utils.py
    https://github.com/s-laine/tempens/blob/master/train.py
"""


class Tap:
    ###################################################################################################
    # Helper class for forking stdout/stderr into a file.
    ###################################################################################################
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ''
        self.file = None
        pass

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        if self.file is not None:
            self.file.write(s)
            self.file.flush()
        else:
            self.buffer = self.buffer + s

    def set_file(self, f):
        assert (self.file is None)
        self.file = f
        self.file.write(self.buffer)
        self.file.flush()
        self.buffer = ''

    def flush(self):
        self.stream.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        self.stream.close()
        if self.file is not None:
            self.file.close()
            self.file = None


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self):
        return {name: meter.val for name, meter in self.meters.items()}
        # return [meter.val for name, meter in self.meters.items()]

    def averages(self):
        return {name: meter.avg for name, meter in self.meters.items()}

    def sums(self):
        return {name: meter.sum for name, meter in self.meters.items()}

    def counts(self):
        return {name: meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el) for el in lst)


def assert_at_most_one(lst):
    assert sum(int(bool(el)) for el in lst) <= 1, ", ".join(str(el) for el in lst)
