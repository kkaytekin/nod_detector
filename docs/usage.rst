User Guide
=========

This guide provides detailed information on using the Nod Detector package.

Configuration
------------

You can configure the detector with various parameters:

.. code-block:: python

   from nod_detector import NodDetector
   
   detector = NodDetector(
       min_confidence=0.5,  # Minimum confidence threshold for detections
       min_nod_duration=0.5,  # Minimum duration of a nod in seconds
       visualize=True,  # Enable visualization
       output_dir="output"  # Directory to save output files
   )

Processing Videos
----------------

Process a video file and get results:

.. code-block:: python

   # Process video and get results
   results = detector.process_video(
       "input.mp4",
       output_video="output.mp4"  # Optional: save processed video
   )
   
   # Results include timestamps and confidence scores
   for i, nod in enumerate(results['nods'], 1):
       print(f"Nod {i}: Start: {nod['start_time']:.2f}s, "
             f"End: {nod['end_time']:.2f}s, "
             f"Confidence: {nod['confidence']:.2f}")

Working with Webcam
------------------

Process video from a webcam in real-time:

.. code-block:: python

   import cv2
   from nod_detector import NodDetector
   
   detector = NodDetector()
   cap = cv2.VideoCapture(0)
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break
           
       # Process frame
       result = detector.process_frame(frame)
       
       # Display results
       cv2.imshow('Nod Detector', result['frame'])
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
   cap.release()
   cv2.destroyAllWindows()

Saving and Loading Results
-------------------------

Save results to a JSON file:

.. code-block:: python

   import json
   
   # Save results
   with open('results.json', 'w') as f:
       json.dump(results, f, indent=2)
   
   # Load results
   with open('results.json', 'r') as f:
       loaded_results = json.load(f)
