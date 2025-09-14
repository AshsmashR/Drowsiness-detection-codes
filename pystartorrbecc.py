from pyorbbecsdk import *
import cv2
import numpy as np

import time
import cv2
import numpy as np


ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20      # 20 mm
MAX_DEPTH = 10000   # 10000 mm

#Creates a Config and a Pipeline object for the camera stream.

#Initializes a TemporalFilter instance for depth smoothing.



class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


def frame_to_bgr_image(color_frame):
    width = color_frame.get_width()
    height = color_frame.get_height()
    data = color_frame.get_data()
    img = np.frombuffer(data, dtype=np.uint8)
    img = img.reshape((height, width, 3))
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def main():
    config = Config()
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)

    try:
        # Enable color stream with requested settings or default
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError:
            color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # Enable depth stream with default profile
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

    except Exception as e:
        print("Error enabling streams:", e)
        return

    pipeline.start(config)
    last_print_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            # Color frame processing
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = frame_to_bgr_image(color_frame)
                cv2.imshow("Color Viewer", color_image)

            # Depth frame processing
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                if depth_frame.get_format() != OBFormat.Y16:
                    print("Unexpected depth format:", depth_frame.get_format())
                    continue

                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()

                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))
                depth_data = depth_data.astype(np.float32) * scale

                # Filter depth values to valid range
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                depth_data = depth_data.astype(np.uint16)

                # Apply temporal smoothing
                depth_data = temporal_filter.process(depth_data)

                # Print center depth occasionally
                center_y, center_x = height // 2, width // 2
                center_depth = depth_data[center_y, center_x]
                current_time = time.time()
                if current_time - last_print_time >= PRINT_INTERVAL:
                    print(f"Center depth: {center_depth:.2f} meters")
                    last_print_time = current_time

                # Normalize and color map for display
                depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("Depth Viewer", depth_vis)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



