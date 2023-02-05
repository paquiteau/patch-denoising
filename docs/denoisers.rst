LLR Denosing methods
=====================

Patch-denoise implemement several local-low-rank methods, based on singular values thresholding.


Singular Value thresholding
---------------------------

General Procedure
~~~~~~~~~~~~~~~~~

Consider a sequence of image or volume. From this image, patches are extracted, processed and recombined.

1. Extraction
^^^^^^^^^^^^^

The extraction of the patch consist of selecting a spatial region of the image, and take all the time information associated to this region.
The patch is then flatten in a 2D Matrix (so called Casorati matrix). A row represents the temporal evolution of a single voxel, and a column is the flatten image at a specific time point.
Moreover, a mask, defining a Region of Interest (ROI) can be used to determined is a patch should be processed or not (and save computations).

.. warning::
    The size of the patch and the overlapping are the main factor for computational cost. Moreover for the SVD-based process to work well, it is required to have a "tall" matrix , i.e. that the number of row is greater than the number of column.

2. Processing
^^^^^^^^^^^^^

Each patch is processed, by applying a threshold function on the centered singular value decomposition of the :math:`M \times N` patch:

.. math::

    X = U S V^T + M

Where :math:`M = \langle X \rangle` is the mean of each row of :math:`X`, and :math:`U,S,V^T` is the SVD decomposition of :math:`X-M`.
In particular, :math:`S=\mathrm{diag}(\sigma_1, \dots, \sigma_n)`

The processing of the singular values by a threshold function :math:`\eta(\sigma_i) = \sigma_i'` yields  new (typically sparser) singular values :math:`S'`

Then the processed patch is defined as:

.. math::

    \hat{X} = U \mathcal{T}(S) V^T + M

3. Recombination
^^^^^^^^^^^^^^^^

The recombination of processed patches uses weights associated to each patch after its processing to determine the final value in case of patch overlapping.
Currently three recombination are considered:

- Identical weights. The patches values for a pixel are averaged together (available with ``recombination='average'``)

- Sparse Patch promotion. The patches values are for a pixel are average with weights :math:`\theta`. This Weighted method comes from [1]_ (available with ``recombination='weighted'``), let :math:`P` the number of patch overlapping for voxel :math:`x_i`, and :math:`\hat{x_i}(p)` the value  associated to each patch. the final value of pixel :math:`\hat{x_i}` is

.. math::
   \hat{x_i} = \frac{\sum_{p=1}^P\theta_p\hat{x_j}(p)}{\sum_{p=1}^P\theta_p} \quad \text{where } \theta_p = \frac{1}{1+\|S'_p\|_0}

The more the processed patch :math:`S'_p` is sparse, the bigger the weight associated to it.

- Use the center of patch. In the case of maximally overlapping patches, the patch center value is use for the corresponding pixel.

.. seealso::
   :class:`~patch_denoise.space_time.base.BaseSpaceTimeDenoiser`
        For the generic patch processing algorithm.

Raw Singular Value Thresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For raw singular value thresholding, the threshold function is simply a hard threshold on the singular value, according to a provided threshold.

.. math::
   \eta_\tau(\sigma_i) = \begin{cases}
   \sigma_i & \text{if}\quad \sigma_i > \tau \\
   0 & \text{otherwise}
   \end{cases}

.. seealso::
   :class:`~patch_denoise.space_time.lowrank.RawSVDDenoiser`
        For the implementation.

MP-PCA Thresholding
~~~~~~~~~~~~~~~~~~~

MP-PCA [2]_ uses the Marshenko-Pastur distribution to find a threshold for each patch. In particular, the noise variance is estimated from the eigen values (squared singular values) and uses to determined the threshold. (See equations 10-12 in reference).


.. seealso::
   :class:`~patch_denoise.space_time.lowrank.MPPCADenoiser`

Hybrid PCA
~~~~~~~~~~

Hybrid-PCA [3]_ uses an a priori spatial distribution of the noise variance, and the singular values are selected such that the discarded one have a mean less or equal to this a priori.

.. seealso::
    :class:`~patch_denoise.space_time.lowrank.HybridPCADenoiser`


NORDIC
~~~~~~

NORDIC [4]_ makes the assumptions that the image noise level is uniform (for instance by pre processing the image and dividing it by an externally available g-map). The threshold is determined by taking the average of  maximum singular value of a set of randomly generated matrix with the dimension as the flattened patch. The uniform noise level must also be provided.


.. seealso::
    :class:`~patch_denoise.space_time.lowrank.NordicDenoiser`

Optimal Thresholding
~~~~~~~~~~~~~~~~~~~~

An optimal thresholding of the singular values [5]_ is also possible associated with a specific norm (Frobenius, nuclear norm or operator norm).

.. seealso::
   :class:`~patch_denoise.space_time.lowrank.OptimalSVDDenoiser`

Adaptive Thresholding
~~~~~~~~~~~~~~~~~~~~~

Extending the possibility of optimal thresholding using SURE in presence of noise variance estimation [6]_.

.. seealso::
   :class:`~patch_denoise.space_time.lowrank.AdaptiveDenoiser`



References
----------
.. [1] Manjón, José V., Pierrick Coupé, Luis Concha, Antonio Buades, D. Louis Collins, and Montserrat Robles. “Diffusion Weighted Image Denoising Using Overcomplete Local PCA.” PLOS ONE 8, no. 9 (September 3, 2013): e73021. https://doi.org/10.1371/journal.pone.0073021.

.. [2] Veraart, Jelle, Dmitry S. Novikov, Daan Christiaens, Benjamin Ades-Aron, Jan Sijbers, and Els Fieremans. “Denoising of Diffusion MRI Using Random Matrix Theory.” NeuroImage 142 (November 15, 2016): 394–406. https://doi.org/10.1016/j.neuroimage.2016.08.016.

.. [3] https://submissions.mirasmart.com/ISMRM2022/Itinerary/Files/PDFFiles/2688.html

.. [4] Moeller, Steen, Pramod Kumar Pisharady, Sudhir Ramanna, Christophe Lenglet, Xiaoping Wu, Logan Dowdle, Essa Yacoub, Kamil Uğurbil, and Mehmet Akçakaya. “NOise Reduction with DIstribution Corrected (NORDIC) PCA in DMRI with Complex-Valued Parameter-Free Locally Low-Rank Processing.” NeuroImage 226 (February 1, 2021): 117539. https://doi.org/10.1016/j.neuroimage.2020.117539.
.. [5] Gavish, Matan, and David L. Donoho. “Optimal Shrinkage of Singular Values.” IEEE Transactions on Information Theory 63, no. 4 (April 2017): 2137–52. https://doi.org/10.1109/TIT.2017.2653801.
.. [6] J. Josse and S. Sardy, “Adaptive Shrinkage of singular values.” arXiv, Nov. 22, 2014. doi: 10.48550/arXiv.1310.6602.
