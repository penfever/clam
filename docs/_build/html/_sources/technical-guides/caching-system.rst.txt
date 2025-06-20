Caching System
==============

Guide to CLAM's intelligent caching system for embeddings and model outputs.

.. note::
   This technical guide is under development.

Overview
--------

CLAM provides caching for:

* TabPFN embeddings
* Audio embeddings (Whisper/CLAP)
* Vision embeddings (DINOV2)
* VLM model outputs

Coming Soon
-----------

This guide will cover:

* Cache configuration and tuning
* Cache invalidation strategies
* Distributed caching
* Performance impact analysis

Interim Resources
-----------------

Basic caching can be enabled with the ``cache_dir`` parameter:

.. code-block:: python

   classifier = ClamTsneClassifier(
       cache_dir="./clam_cache"
   )