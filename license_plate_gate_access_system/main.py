import concurrent.futures
import cv2
import imageProcessing as imgprocess
import os
import time
import traceback


CAMERA_FORMAT = "RGB888"
CAMERA_RESOLUTION = (2592, 1944)
USB_CAMERA_RESOLUTION = (
    int(os.environ.get("PLAKA_USB_WIDTH", "1280")),
    int(os.environ.get("PLAKA_USB_HEIGHT", "720")),
)
FRAME_CAPTURE_INTERVAL_SECONDS = float(os.environ.get("PLAKA_FRAME_INTERVAL", "0.5"))
ERROR_LOG_INTERVAL_SECONDS = 5
USB_CAMERA_RESCAN_SECONDS = float(os.environ.get("PLAKA_USB_RESCAN_SECONDS", "5"))
USB_CAMERA_INDEX_LIMIT = int(os.environ.get("PLAKA_USB_INDEX_LIMIT", "10"))
USB_CAMERA_READ_FAILURE_LIMIT = int(os.environ.get("PLAKA_USB_READ_FAILURE_LIMIT", "3"))
SUCCESS_RESULT_PREFIX = "plaka kaydedildi: "
CAMERA_CONTROLS = {
    "FrameRate": float(os.environ.get("PLAKA_CAMERA_FPS", "15.0")),
    "AeEnable": True,
    "AwbEnable": True,
}


class CameraReadError(Exception):
    pass


class UsbCameraSource:
    kind = "usb"

    def __init__(self, index, capture):
        self.index = index
        self.name = f"usb{index}"
        self.capture = capture
        self.read_failures = 0

    def capture_frame(self):
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise CameraReadError(f"{self.name} goruntu veremedi")
        self.read_failures = 0
        return frame

    def close(self):
        self.capture.release()


class PiCameraSource:
    kind = "picamera"
    name = "picamera"

    def __init__(self, camera):
        self.camera = camera

    def capture_frame(self):
        frame = self.camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self):
        self.camera.stop()
        self.camera.close()


def main():
    imgprocess.prepare_debug_output_dir()
    sources = {}
    source_failures = {}
    last_error_log_at = 0.0
    last_usb_scan_at = 0.0

    try:
        sources = start_available_cameras()
        while True:
            frame_started_at = time.monotonic()

            if should_rescan_usb(sources, last_usb_scan_at):
                last_usb_scan_at = time.monotonic()
                sources = refresh_usb_sources(sources)
                if not sources:
                    sources = start_picamera_fallback()

            if not sources:
                last_error_log_at = log_message_if_due(
                    last_error_log_at,
                    "Calisan USB kamera veya PiCamera bulunamadi.",
                )
                wait_for_next_frame(frame_started_at)
                continue

            try:
                source_results = process_sources(sources.values())
                for source_name, results in source_results:
                    for result in results:
                        handle_result(source_name, result, source_failures)
            except Exception:
                last_error_log_at = log_exception_if_due(last_error_log_at)

            remove_failed_usb_sources(sources)
            if not sources:
                sources = start_picamera_fallback()
            wait_for_next_frame(frame_started_at)
    finally:
        close_sources(sources.values())


def start_available_cameras():
    usb_sources = discover_usb_sources()
    if usb_sources:
        print(f"USB kamera modu: {', '.join(sorted(usb_sources))}", flush=True)
        return usb_sources
    return start_picamera_fallback()


def start_picamera_fallback():
    try:
        source = start_picamera()
    except Exception:
        print("PiCamera baslatilamadi:", traceback.format_exc().strip(), flush=True)
        return {}
    return {source.name: source}


def start_picamera():
    from picamera2 import Picamera2

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
        log_picamera_started(camera)
        return PiCameraSource(camera)
    except Exception:
        try:
            camera.stop()
        except Exception:
            pass
        camera.close()
        raise


def discover_usb_sources(existing_sources=None):
    existing_sources = existing_sources or {}
    sources = {}
    for index in range(USB_CAMERA_INDEX_LIMIT):
        name = f"usb{index}"
        if name in existing_sources:
            sources[name] = existing_sources[name]
            continue

        source = open_usb_source(index)
        if source:
            sources[source.name] = source
    return sources


def open_usb_source(index):
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        return None

    width, height = USB_CAMERA_RESOLUTION
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        capture.release()
        return None

    source = UsbCameraSource(index, capture)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"USB kamera baslatildi: {source.name}, {actual_width}x{actual_height}", flush=True)
    return source


def should_rescan_usb(sources, last_usb_scan_at):
    return time.monotonic() - last_usb_scan_at >= USB_CAMERA_RESCAN_SECONDS


def refresh_usb_sources(sources):
    if sources and not using_usb_sources(sources):
        usb_sources = discover_usb_sources()
        if not usb_sources:
            return sources
        close_sources(sources.values())
        print(f"USB kamera modu: {', '.join(sorted(usb_sources))}", flush=True)
        return usb_sources

    current = discover_usb_sources(sources)
    removed_names = sorted(set(sources) - set(current))
    for name in removed_names:
        sources[name].close()
        print(f"Kamera devreden cikti: {name}", flush=True)
    return current


def using_usb_sources(sources):
    return any(source.kind == "usb" for source in sources.values())


def process_sources(sources):
    source_list = list(sources)
    max_workers = max(1, len(source_list))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {
            executor.submit(process_source, source): source
            for source in source_list
        }
        for future in concurrent.futures.as_completed(future_to_source):
            source = future_to_source[future]
            try:
                results.append((source.name, future.result()))
            except CameraReadError as exc:
                source.read_failures += 1
                print(f"{exc} ({source.read_failures}/{USB_CAMERA_READ_FAILURE_LIMIT})", flush=True)
            except Exception:
                print(f"{source.name} isleme hatasi:", traceback.format_exc().strip(), flush=True)
    return results


def process_source(source):
    frame = source.capture_frame()
    return imgprocess.rec_all(frame)


def remove_failed_usb_sources(sources):
    failed_names = [
        name
        for name, source in sources.items()
        if source.kind == "usb" and source.read_failures >= USB_CAMERA_READ_FAILURE_LIMIT
    ]
    for name in failed_names:
        sources[name].close()
        del sources[name]
        print(f"Kamera devreden cikti: {name}", flush=True)


def close_sources(sources):
    for source in sources:
        try:
            source.close()
        except Exception:
            pass


def log_picamera_started(camera):
    configuration = camera.camera_configuration()
    main_config = configuration.get("main", {})
    width, height = main_config.get("size", CAMERA_RESOLUTION)
    pixel_format = main_config.get("format", CAMERA_FORMAT)
    print(
        f"PiCamera baslatildi: {width}x{height}, format={pixel_format}",
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


def handle_result(source_name, result, source_failures):
    if is_failure(result) and has_debug_snapshot(result):
        source_failures[source_name] = source_failures.get(source_name, 0) + 1
        log_failure_count(source_name, source_failures[source_name])
    elif is_failure(result):
        return
    else:
        source_failures[source_name] = 0
        log_success(source_name, result)


def is_failure(result):
    return result and not result.startswith("plaka ")


def has_debug_snapshot(result):
    return " | debug=" in result


def log_failure_count(source_name, failure_count):
    print(f"{source_name} basarisiz deneme sayisi: {failure_count}", flush=True)


def log_exception_if_due(last_error_log_at):
    now = time.monotonic()
    if last_error_log_at and now - last_error_log_at < ERROR_LOG_INTERVAL_SECONDS:
        return last_error_log_at

    print("Son hata:", traceback.format_exc().strip(), flush=True)
    return now


def log_message_if_due(last_error_log_at, message):
    now = time.monotonic()
    if last_error_log_at and now - last_error_log_at < ERROR_LOG_INTERVAL_SECONDS:
        return last_error_log_at
    print(message, flush=True)
    return now


def log_success(source_name, result):
    plate = get_plate_from_success_result(result)
    if plate:
        print(f"{source_name}: {plate}", flush=True)


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
