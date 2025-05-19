Examples
========

This page contains practical examples of using the Nod Detector package.

Basic Example
------------

Process a video file and visualize the results:

.. code-block:: python

   from nod_detector import NodDetector
   
   # Initialize with visualization enabled
   detector = NodDetector(visualize=True)
   
   # Process video
   results = detector.process_video(
       "input.mp4",
       output_video="output.mp4"
   )
   
   # Print summary
   print(f"Processed {len(results['frames'])} frames")
   print(f"Detected {len(results['nods'])} nods")

Real-time Webcam Processing
--------------------------

Process video from your webcam in real-time:

.. code-block:: python

   import cv2
   from nod_detector import NodDetector
   
   detector = NodDetector(visualize=True)
   cap = cv2.VideoCapture(0)
   
   print("Press 'q' to quit")
   while True:
       ret, frame = cap.read()
       if not ret:
           break
           
       # Process frame
       result = detector.process_frame(frame)
       
       # Display FPS
       fps = detector.get_fps()
       cv2.putText(result['frame'], f"FPS: {fps:.1f}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
       # Display results
       cv2.imshow('Nod Detector', result['frame'])
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
   cap.release()
   cv2.destroyAllWindows()

Batch Processing
---------------

Process multiple videos in a directory:

.. code-block:: python

   from pathlib import Path
   from nod_detector import NodDetector
   
   detector = NodDetector()
   input_dir = Path("videos")
   output_dir = Path("results")
   output_dir.mkdir(exist_ok=True)
   
   # Process all MP4 files in the directory
   for video_path in input_dir.glob("*.mp4"):
       print(f"Processing {video_path.name}...")
       results = detector.process_video(
           str(video_path),
           output_video=str(output_dir / f"processed_{video_path.name}")
       )
       
       # Save results
       output_json = output_dir / f"{video_path.stem}_results.json"
       with open(output_json, 'w') as f:
           json.dump(results, f, indent=2)
       
       print(f"Saved results to {output_json}")

Custom Callback
--------------

Use a callback function to process each frame:

.. code-block:: python

   def on_frame_processed(frame_result):
       """Custom callback for each processed frame."""
       frame = frame_result['frame']
       if frame_result['detected']:
           print(f"Nod detected at frame {frame_result['frame_number']}")
       return frame
   
   # Initialize with custom callback
   detector = NodDetector(frame_callback=on_frame_processed)
   results = detector.process_video("input.mp4")
