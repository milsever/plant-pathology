"""
Refs:
    https://github.com/s-laine/tempens/blob/master/report.py
"""

import datetime
import glob
import os
import shutil
import sys
import socket
import getpass
from collections import OrderedDict


def create_result_subdir(result_dir, run_desc):
    ordinal = 0
    for fname in glob.glob(os.path.join(result_dir, '*')):
        try:
            fbase = os.path.basename(fname)
            ford = int(fbase[:fbase.find('-')])
            ordinal = max(ordinal, ford + 1)
        except ValueError:
            pass

    result_subdir = os.path.join(result_dir, '%03d-%s' % (ordinal, run_desc))
    if os.path.isdir(result_subdir):
        return create_result_subdir(result_dir, run_desc)  # Retry.
    if not os.path.isdir(result_subdir):
        os.makedirs(result_subdir)
    return result_subdir


def export_sources(target_dir):
    os.makedirs(target_dir)
    for ext in ('py', 'pyproj', 'sln'):
        for fn in glob.glob('*.' + ext):
            shutil.copy2(fn, target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src', '*.' + ext)):
                shutil.copy2(fn, target_dir)


def export_run_details(fname, args):
    with open(fname, 'wt') as f:
        f.write('%-16s%s\n' % ('Host', socket.gethostname().lower()))
        f.write('%-16s%s\n' % ('User', getpass.getuser()))
        f.write('%-16s%s\n' % ('Date', datetime.datetime.today()))
        f.write('%-16s%s\n' % ('CUDA device', args.cuda_device_num))
        f.write('%-16s%s\n' % ('Working dir', os.getcwd()))
        f.write('%-16s%s\n' % ('Executable', sys.argv[0]))
        f.write('%-16s%s\n' % ('Arguments', ' '.join(sys.argv[1:])))


def export_config(fname, args):
    with open(fname, 'wt') as fout:
        for key in sorted(args.__dict__):
            fout.write("{}: {}\n".format(key, args.__dict__[key]))


class GenericCSV(object):
    def __init__(self, fname, *fields):
        self.fields = fields
        self.fout = open(fname, 'wt')
        self.fout.write(','.join(fields) + '\n')
        self.fout.flush()

    # def add_data(self, *values):
    def add_data(self, values):
        assert len(values) >= len(self.fields)
        # strings = [v if isinstance(v, str) else '%g' % v for v in values]
        strings = [values[f] if isinstance(values[f], str) else '%g' % values[f] for f in self.fields]
        self.fout.write(','.join(strings) + '\n')
        self.fout.flush()

    def close(self):
        self.fout.close()

    def __enter__(self):  # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.close()
