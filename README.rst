========================
Patch Denoising Methods
========================
.. list-table:: 
   :widths: 25 25 25
   :header-rows: 0

   * - .. image:: https://img.shields.io/codecov/c/github/paquiteau/patch-denoising
        :target: https://app.codecov.io/gh/paquiteau/patch-denoising
     - .. image:: https://github.com/paquiteau/patch-denoising/workflows/CI/badge.svg
     - .. image:: https://github.com/paquiteau/patch-denoising/workflows/CD/badge.svg
   * - .. image:: https://img.shields.io/badge/style-black-black
     - .. image:: https://img.shields.io/badge/docs-Sphinx-blue
        :target: https://paquiteau.github.io/patch-denoising
     - .. image:: https://img.shields.io/pypi/v/patch-denoise
        :target: https://pypi.org/project/patch-denoise/


This repository implements patch-denoising methods, with a particular focus on local-low rank methods.

The target application is functional MRI thermal noise removal, but this methods can be applied to a wide range of image modalities.

It includes several local-low-rank based denoising methods:

1. MP-PCA
2. Hybrid-PCA
3. NORDIC
4. Optimal Thresholding
5. Raw Singular Value Thresholding

A mathematical description of theses methods is available in the documentation.



Installation
============

patch-denoise requires Python>=3.8


Quickstart
==========

After installing you can use the ``patch-denoise`` command-line.

.. code::

   $ patch-denoise input_file.nii output_file.nii --mask="auto"

See ``patch-denoise --help`` for detailled options.

Documentation and Examples
==========================

Documentation and examples are available at https://paquiteau.github.io/patch-denoising/


Development version
===================

.. code::

   $ git clone https://github.com/paquiteau/patch-denoising
   $ pip install -e patch-denoising[dev,doc,test,optional]
