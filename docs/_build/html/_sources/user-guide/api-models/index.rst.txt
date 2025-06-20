API Models Integration Guide
===============================

Guide for integrating CLAM with commercial API models.

.. note::
   This section is under development. Please refer to the examples in the meantime.

Overview
--------

CLAM supports integration with:

* OpenAI GPT-4V and GPT-4o models
* Google Gemini Vision models
* Anthropic Claude models (future support)

Quick Start
-----------

OpenAI Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   os.environ["OPENAI_API_KEY"] = "your-api-key"

   from clam.models.clam_tsne import ClamTsneClassifier

   classifier = ClamTsneClassifier(
       modality="vision",
       openai_model="gpt-4o",
       enable_thinking=True
   )

Google Gemini Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   os.environ["GOOGLE_API_KEY"] = "your-api-key"

   classifier = ClamTsneClassifier(
       modality="vision",
       gemini_model="gemini-2.0-flash-exp"
   )

Examples
--------

See ``examples/vision/`` directory for API model examples.

API Reference
-------------

See :doc:`../../api-reference/clam.models` for detailed API documentation.