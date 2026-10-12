GPU-Accelerated Denoising
=========================

For large datasets, ``patch_denoise.gpu`` provides a PyTorch/Triton implementation of the
same local-low-rank denoisers, operating on batches of patches directly on the GPU.

It is an optional extra, installed with:

.. code-block:: bash

    pip install patch-denoise[gpu]

and enabled from the command line with the ``--gpu`` flag (see :doc:`usage`), or used
programmatically:

.. seealso::
   :func:`~patch_denoise.gpu.main.main_gpu`
        Entry point running the full GPU denoising pipeline on a NIfTI file.
   :func:`~patch_denoise.gpu.main.make_denoiser`
        Instantiate the appropriate GPU denoiser module for a given method.

Denoisers
---------

The GPU denoisers are implemented as ``torch.nn.Module``, operating on batches of patches
of shape ``(B, *patch_shape)``.

.. seealso::
   :class:`~patch_denoise.gpu.denoiser.OptimalSVDDenoiser`
        GPU counterpart of :class:`~patch_denoise.space_time.lowrank.OptimalSVDDenoiser`.
   :class:`~patch_denoise.gpu.denoiser.MPPCADenoiser`
        GPU counterpart of :class:`~patch_denoise.space_time.lowrank.MPPCADenoiser`.

Patch extraction and recombination
-----------------------------------

Patches are extracted and recombined with dedicated Triton kernels, exposed through:

.. seealso::
   :class:`~patch_denoise.gpu.dataloader.PatchDataset`
        Iterates over the patches of a volume, in batches.
   :func:`~patch_denoise.gpu.dataloader.patchify_tensor`
        Extract overlapping patches from a tensor.

Batch size autotuning
----------------------

The GPU pipeline can automatically pick a batch size fitting in the available GPU memory.

.. seealso::
   :func:`~patch_denoise.gpu.autotune.autotune_batch_size`
        Find (and cache) the largest working batch size for a given denoiser and patch shape.
