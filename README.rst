===============================
Patch Denoising Methods
===============================

This repository implements patch-denoising methods, with a particular focus on local-low rank methods.

The target application is functional MRI thermal noise removal, but this methods can be applied to a wide range of image modalities.

It includes several local-low-rank based denoising methods:
1. MP-PCA
2. Hybrid-PCA
3. NORDIC
4. Optimal Thresholding
5. Raw Singular Value Thresholding

A mathematical description of theses methods will be available in the documentation.



Installation
================

Development version
-------------------

.. code::

   git clone https://github.com/paquiteau/patch-denoising
   pip install -e patch-denoising[dev, test]



Quickstart
==============

To check if everything worked fine you can run:

.. code::

   python -c 'import denoiser'




Examples
==========

Examples are available in the documentation.
