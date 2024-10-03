#########################################
Using patch-denoise from the command line
#########################################

``patch-denoise`` minimally requires a path to a NIfTI file,
but it can take advantage of reconstructed phase data and/or noise volumes.

.. argparse::
   :ref: patch_denoise.bindings.cli._get_parser
   :prog: patch-denoise
   :func: _get_parser
