import cv2
import imageProcessing as imgprocess
import os
import time
import traceback
from picamera2 import Picamera2


CAMERA_FORMAT = "RGB888"
CAMERA_RESOLUTION = (2592, 1944)
FRAME_CAPTURE_INTERVAL_SECONDS = 0.5
ERROR_LOG_INTERVAL_SECONDS = 5
SUCCESS_RESULT_PREFIX = "plaka kaydedildi: "
CAMERA_CONTROLS = {
    "FrameRate": float(os.environ.get("PLAKA_CAMERA_FPS", "15.0")),
    "AeEnable": True,
    "AwbEnable": True,
}


def start_camera():
    camera = Picamera2()
    try:
        config = camera.create_video_configuration(
            main={"format": CAMERA_FORMAT, "size": CAMERA_RESOLUTION},
            controls=CAMERA_CONTROLS,
            buffer_count=2,
        )
        camera.configure(config)
        camera.start()
        apply_optional_camera_controls(camera)
        log_camera_started(camera)
        return camera
    except Exception:
        try:
            camera.stop()
        except Exception:
            pass
        camera.close()
        raise


def main():
    imgprocess.prepare_debug_output_dir()
    camera = start_camera()
    failure_count = 0
    failure_line_active = False
    last_error_log_at = 0.0

    try:
        while True:
            frame_started_at = time.monotonic()
            try:
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                result = imgprocess.rec(frame)
                if is_failure(result) and has_debug_snapshot(result):
                    failure_count += 1
                    log_failure_count(failure_count)
                    failure_line_active = True
                elif is_failure(result):
                    continue
                else:
                    if failure_line_active:
                        print(flush=True)
                    log_success(result)
                    failure_count = 0
                    failure_line_active = False
            except Exception:
                if failure_line_active:
                    print(flush=True)
                    failure_line_active = False
                last_error_log_at = log_exception_if_due(last_error_log_at)
            wait_for_next_frame(frame_started_at)
    finally:
        if failure_line_active:
            print(flush=True)
        camera.stop()
        camera.close()


def log_camera_started(camera):
    configuration = camera.camera_configuration()
    main_config = configuration.get("main", {})
    width, height = main_config.get("size", CAMERA_RESOLUTION)
    pixel_format = main_config.get("format", CAMERA_FORMAT)
    print(
        f"Kamera baslatildi: {width}x{height}, format={pixel_format}",
        flush=True,
    )


def apply_optional_camera_controls(camera):
    for controls in (
        {"AfMode": 2},
        {"AfSpeed": 1},
    ):
        try:
            camera.set_controls(controls)
        except Exception:
            pass


def is_failure(result):
    return result and not result.startswith("plaka ")


def has_debug_snapshot(result):
    return " | debug=" in result


def log_failure_count(failure_count):
    print(f"\rBasarisiz deneme sayisi: {failure_count}", end="", flush=True)


def log_exception_if_due(last_error_log_at):
    now = time.monotonic()
    if last_error_log_at and now - last_error_log_at < ERROR_LOG_INTERVAL_SECONDS:
        return last_error_log_at

    print(flush=True)
    print("Son hata:", traceback.format_exc().strip(), flush=True)
    return now


def log_success(result):
    plate = get_plate_from_success_result(result)
    if plate:
        print(plate, flush=True)


def get_plate_from_success_result(result):
    if not result or not result.startswith(SUCCESS_RESULT_PREFIX):
        return None
    return result[len(SUCCESS_RESULT_PREFIX):].split(" | ", 1)[0]


def wait_for_next_frame(frame_started_at):
    elapsed = time.monotonic() - frame_started_at
    remaining = FRAME_CAPTURE_INTERVAL_SECONDS - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    main()
