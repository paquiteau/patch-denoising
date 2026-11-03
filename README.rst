========================
Patch Denoising Methods
========================


|COVERAGE| |CI| |CD| 
|DOC| |RELEASE|  |PYVERSION|

|LINTER| |STYLE| |LICENSE| |CITATION|


.. |COVERAGE| image:: https://img.shields.io/codecov/c/github/paquiteau/patch-denoising
   :target: https://app.codecov.io/gh/paquiteau/patch-denoising
.. |CI| image:: https://github.com/paquiteau/patch-denoising/workflows/CI/badge.svg
.. |CD| image:: https://github.com/paquiteau/patch-denoising/workflows/CD/badge.svg
.. |LICENSE| image:: https://img.shields.io/github/license/paquiteau/patch-denoising
.. |DOC| image:: https://img.shields.io/badge/docs-Sphinx-blue
  :target: https://paquiteau.github.io/patch-denoising
.. |RELEASE| image:: https://img.shields.io/pypi/v/patch-denoise
   :target: https://pypi.org/project/patch-denoise/
.. |STYLE| image:: https://img.shields.io/badge/style-black-black
   :target: https://github.com/psf/black
.. |LINTER| image:: https://img.shields.io/badge/linter-ruff-inactive
   :target: https://github.com/charliemarsh/ruff
.. |PYVERSION| image:: https://img.shields.io/pypi/pyversions/patch-denoise
   :target: https://pypi.org/project/patch-denoise/
.. |CITATION| image:: https://img.shields.io/badge/paper-hal--openaccess-green
   :target: https://hal.science/hal-03895194
   
This repository implements patch-denoising methods, with a particular focus on local-low rank methods.

The target application is functional MRI thermal noise removal, but this methods can be applied to a wide range of image modalities.

It includes several local-low-rank based denoising methods (see the `documentation <https://paquiteau.github.io/patch-denoising>`_ for more details):

1. MP-PCA
2. Hybrid-PCA
3. NORDIC
4. Optimal Thresholding
5. Raw Singular Value Thresholding

A mathematical description of these methods is available in the documentation.



Installation
============

.. code::

   $ pip install patch-denoise

patch-denoise requires Python>=3.12


Apptainer / Singularity
-----------------------

A container recipe with the GPU extras (PyTorch, Triton) is provided in ``patch-denoise.def``.
Build it once, from a checkout of this repository (no root needed with ``--fakeroot``):

.. code::

   $ apptainer build --fakeroot --build-arg VERSION=$(git describe --tags | sed s/^v//) \
         patch-denoise.sif patch-denoise.def

and run it like the regular command-line. Use ``--nv`` to expose the host GPU, and ``--bind`` for
data that lives outside your home directory:

.. code::

   $ apptainer run --nv --bind /path/to/data patch-denoise.sif input.nii output.nii --mask="auto"

The image bundles CUDA 13 user-space libraries, so the host NVIDIA driver must support CUDA 13
(``nvidia-smi`` reporting driver >= 580). Without ``--nv`` the CPU implementation is used.

On a Slurm cluster:

.. code::

   $ srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
         apptainer run --nv --bind /path/to/data patch-denoise.sif input.nii output.nii --mask="auto"

Quickstart
==========

After installing you can use the ``patch-denoise`` command-line.

.. code::

   $ patch-denoise input_file.nii output_file.nii --mask="auto"

See ``patch-denoise --help`` for detailed options.

Documentation and Examples
==========================

Documentation and examples are available at https://paquiteau.github.io/patch-denoising/


Development version
===================

.. code::

   $ git clone https://github.com/paquiteau/patch-denoising
   $ pip install -e patch-denoising[optional] --group dev
   
Citation
========

If you use this package for academic work, please cite the associated publication, available on `HAL <https://hal.science/hal-03895194>`_ ::

      @inproceedings{comby2023,
        TITLE = {{Denoising of fMRI volumes using local low rank methods}},
        AUTHOR = {Pierre-Antoine, Comby and Zaineb, Amor and Alexandre, Vignaud and Philippe, Ciuciu},
        URL = {https://hal.science/hal-03895194},
        BOOKTITLE = {{ISBI 2023 - International Symposium on Biomedical Imaging 2023}},
        ADDRESS = {Carthagena de India, Colombia},
        YEAR = {2023},
        MONTH = Apr,
        KEYWORDS = {functional MRI ; patch denoising ; singular value thresholding ; functional MRI patch denoising singular value thresholding},
        PDF = {https://hal.science/hal-03895194/file/isbi2023_denoise.pdf},
        HAL_ID = {hal-03895194},
        HAL_VERSION = {v1},
      }


Related Packages
================

- https://github.com/paquiteau/retino-pypeline

  For the application of the denoising in an fMRI pipeline using Nipype
