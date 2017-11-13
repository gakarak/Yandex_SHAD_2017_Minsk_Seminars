#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataKnum = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dataAccP = [83.60, 86.71, 88.47, 89.46, 91.71, 92.75, 94.01]

    plt.plot(dataKnum, dataAccP, '-o')
    plt.grid(True)
    plt.show()