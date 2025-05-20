Quick Start
==========

Get started with Nod Detector in a few simple steps.

Installation
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/kkaytekin/nod_detector.git
   cd nod_detector

   # Create and activate a virtual environment (recommended)
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # On Unix/macOS: source venv/bin/activate

   # Install the package in development mode with all dependencies
   pip install -e .

Basic Usage
-----------

.. code-block:: python

   from nod_detector import NodDetector

   # Initialize the detector
   detector = NodDetector()

   # Process a video file
   results = detector.process_video("input_video.mp4")

   # Get nod detection results
   print(f"Detected {len(results['nods'])} nods in the video")

Command Line Interface
---------------------

Process a video file from the command line:

.. code-block:: bash

   python -m nod_detector --input input_video.mp4 --output results.json

Next Steps
----------
- Learn more in the :doc:`usage` guide
- Check out the :doc:`examples` for advanced usage
- Explore the :doc:`api` reference for detailed information
