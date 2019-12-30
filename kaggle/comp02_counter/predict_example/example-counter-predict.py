#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
import torch
import tensorflow as tf


def main_run(path_idx: str, model_dir: str):
    data_idx = pd.read_csv(path_idx)
    wdir = os.path.dirname(path_idx)
    data_idx['count'] = np.random.randint(0, 1000, len(data_idx))
    # ... model inference code
    #
    data_idx.to_csv(path_idx, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=str, required=True, help='path to index file')
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='directory with model and model-data')
    args = parser.parse_args()
    logging.info('args = {}'.format(args))
    main_run(
        path_idx=args.idx,
        model_dir=args.model_dir
    )
