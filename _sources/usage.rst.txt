############
Command line
############

``patch-denoise`` minimally requires a path to a NIfTI file,
but it can take advantage of reconstructed phase data and/or noise volumes.

.. argparse::
   :ref: patch_denoise.bindings.cli._get_parser
   :prog: patch-denoise
   :func: _get_parser


================================
Using patch-denoise on BIDS data
================================

.. warning::
    These examples assume that the phase data are in radians.
    If they are in arbitrary units,
    you will need to rescale them before running patch-denoise.


Magnitude only
==============

.. code-block:: bash

    patch-denoise \
        sub-01/func/sub-01_task-rest_part-mag_bold.nii.gz \
        sub-01_task-rest_part-mag_desc-denoised_bold.nii.gz \
        --mask auto \
        --method optimal-fro \
        --patch-shape 11 \
        --patch-overlap 5 \
        --recombination weighted \
        --mask-threshold 1


Magnitude with noise volumes
============================

.. code-block:: bash

    patch-denoise \
        sub-01/func/sub-01_task-rest_part-mag_bold.nii.gz \
        sub-01_task-rest_part-mag_desc-denoised_bold.nii.gz \
        --noise-map sub-01/func/sub-01_task-rest_part-mag_noRF.nii.gz \
        --mask auto \
        --method optimal-fro \
        --patch-shape 11 \
        --patch-overlap 5 \
        --recombination weighted \
        --mask-threshold 1


Magnitude and phase
===================

.. code-block:: bash

    patch-denoise \
        sub-01/func/sub-01_task-rest_part-mag_bold.nii.gz \
        sub-01_task-rest_part-mag_desc-denoised_bold.nii.gz \
        --input-phase sub-01/func/sub-01_task-rest_part-phase_bold.nii.gz \
        --mask auto \
        --method optimal-fro \
        --patch-shape 11 \
        --patch-overlap 5 \
        --recombination weighted \
        --mask-threshold 1


Magnitude and phase with noise volumes
======================================

.. code-block:: bash

    patch-denoise \
        sub-01/func/sub-01_task-rest_part-mag_bold.nii.gz \
        sub-01_task-rest_part-mag_desc-denoised_bold.nii.gz \
        --input-phase sub-01/func/sub-01_task-rest_part-phase_bold.nii.gz \
        --noise-map sub-01/func/sub-01_task-rest_part-mag_noRF.nii.gz \
        --noise-map-phase sub-01/func/sub-01_task-rest_part-phase_noRF.nii.gz \
        --mask auto \
        --method optimal-fro \
        --patch-shape 11 \
        --patch-overlap 5 \
        --recombination weighted \
        --mask-threshold 1
