# -*- coding: utf-8 -*-
"""
Experimental Data denoising
===========================

This is a example script to test various denoising methods on real-word fMRI data.

Source data should a sequence of 2D or 3D data, the temporal dimension being the last one.

The available denoising methods are "nordic", "mp-pca", "hybrid-pca", "opt-fro", "opt-nuc" and "opt-op".

"""
import numpy as np
import matplotlib.pyplot as plt
