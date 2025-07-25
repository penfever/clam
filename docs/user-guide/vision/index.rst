Vision Classification Guide
============================

Comprehensive guide for image classification with CLAM.

.. note::
   This section is under development. Please refer to the examples in the meantime.

Overview
--------

CLAM supports image classification through:

* DINOV2 embeddings for feature extraction
* t-SNE/PCA visualization of image embeddings
* Vision Language Model classification
* Support for CIFAR-10/100, ImageNet, and custom datasets

Quick Start
-----------

.. code-block:: bash

   # Test CIFAR-10 classification
   python examples/vision/evaluate_all_vision.py \\
       --datasets cifar10 \\
       --models clam_tsne \\
       --quick_test

Examples
--------

See ``examples/vision/`` directory for complete examples.

API Reference
-------------

See :doc:`../../api-reference/clam.models` for detailed API documentation.