from pathlib import Path
from ultralytics import YOLO
import openvino as ov
import cv2
import collections
import time
import numpy as np
from utils import VideoPlayer
import sys

det_model_path = Path(r"best_openvino_model\best.xml")
core = ov.Core()
det_ov_model = core.read_model(det_model_path)
det_model = YOLO(det_model_path.parent, task="detect")
input=Path(r"test_input.avi")
device = sys.argv[1]

# Main processing function to run object detection.
def run_object_detection(
    source=0,    
    skip_first_frames=0,    
    device=device,
    video_width: int = None,  # if not set the original size is used
):
    player = None
    ov_config = {}
    
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    compiled_model = core.compile_model(det_ov_model, device, ov_config)

    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)

    det_model.predictor.model.ov_compiled_model = compiled_model

    try:
        # Create a video player to play with target fps.
        player = VideoPlayer(source=source, flip=False, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing.
        player.start()        
        title = "Press ESC to Exit"
        cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break

            if video_width:
                # If the frame is larger than video_width, reduce size to improve the performance.
                # If more, increase size for better demo expirience.

                scale = video_width / max(frame.shape)
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Get the results.
            input_image = np.array(frame)

            start_time = time.time()
            detections = det_model(input_image, verbose=False)
            stop_time = time.time()
            frame = detections[0].plot()

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            # Use this workaround if there is flickering.
            cv2.imshow(winname=title, mat=frame)
            key = cv2.waitKey(1)
            # escape = 27
            if key == 27:
                break
            
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        cv2.destroyAllWindows()

run_object_detection(source=input, device=device)
