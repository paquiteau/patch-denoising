Denoiser descriptions
=====================


Patch-denoiser is a python implementation of different local low rank method for denoising, targeting fMRI thermal noise removal (ie 4D data)


It includes the current denoising methods.

- Singular Value Thresholding methods:

    1. Raw singular Value Thresholding
    2. MP-PCA
    3. Hybrid-PCA
    4. NORDIC
    5. Optimal Thresholding

- Image domain methods (TBA):

    1. BM4D
    2. Non-Local Mean


Singular Value thresholding methods
-----------------------------------

General Procedure
~~~~~~~~~~~~~~~~~

Consider a sequence of image or volume. From this image, patches are extracted, processed (ie denoised) and recombined.

1. Extraction

   The extraction of the patch consist of selecting a spatial region of the image, and take all the time information associated to this region.
   The patch is then flatten in a 2D Matrix (so called Casorati matrix). A row represents the temporal evolution of a single pixel/voxel, and a column is the flatten image at a specific time point.
   Moreover, a mask, defining a Region of Interest (ROI) can be used to determined is a patch should be processed or not (and save computations).

   .. note::
      The size of the patch and the overlapping are the main factor for computational cost. Moreover for the SVD-based process to work well, it is required to have a "tall" matrix , ie that the number of row is greater than the number of column.

2. Processing

   Each patch is processed, by applying a threshold function on the centered singular value decomposition of the :math:`M \times N` patch:

   .. math::

      X = U S V^T + M

   Where :math:`M = \langle X \rangle` is the mean of each row of :math:`X`, and :math:`U,S,V^T` is the SVD decomposition of :math:`X-M`.
   In particular, :math:`S=\diag(\sigma_1, \dots, \sigma_n)`


   The threshold function is :math:`\mathcal{T}(S) = S'` where :math:`S'` is typically sparser than :math:`S`.

   Then the processed patch is defined as:

   .. math::

      \hat{X} = U \mathcal{T}(S) V^T + M

3. Recombination
   The recombination of processed patches uses weights associated to each patch after its processing to determine the final value in case of patch overlapping.
   Currently three recombination are considered:

   - Identical weights. The patches values for a pixel are averaged together (available with ``recombination='average'``)

   - Sparse Patch promotion. The patches values are for a pixel are average with weights :math:`\theta`. This Weighted method comes from Manjon2013 (available with ``recombination='weighted'``)

  - Use the center of patch. In the case of maximally overlapping patches, the patch center value is use for the corresponding pixel.


Raw Singular Value Thresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MP-PCA Thresholding
~~~~~~~~~~~~~~~~~~~

Hybrid PCA
~~~~~~~~~~

NORDIC
~~~~~~

Optimal Thresholding
~~~~~~~~~~~~~~~~~~~~
