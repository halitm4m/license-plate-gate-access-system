import concurrent.futures
import logging
import os

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
logging.disable(logging.CRITICAL)

import cv2
import imageProcessing as imgprocess
import save
import time
from pathlib import Path


CAMERA_FORMAT = "RGB888"
CAMERA_RESOLUTION = (2592, 1944)
USB_CAMERA_RESOLUTION = (
    int(os.environ.get("PLAKA_USB_WIDTH", "1280")),
    int(os.environ.get("PLAKA_USB_HEIGHT", "720")),
)
FRAME_CAPTURE_INTERVAL_SECONDS = float(os.environ.get("PLAKA_FRAME_INTERVAL", "0.5"))
USB_CAMERA_RESCAN_SECONDS = float(os.environ.get("PLAKA_USB_RESCAN_SECONDS", "5"))
USB_CAMERA_INDEX_LIMIT = int(os.environ.get("PLAKA_USB_INDEX_LIMIT", "10"))
USB_CAMERA_READ_FAILURE_LIMIT = int(os.environ.get("PLAKA_USB_READ_FAILURE_LIMIT", "3"))
SUCCESS_RESULT_PREFIX = "plaka okundu: "
WHITELIST_FILE = Path(os.environ.get(
    "PLAKA_WHITELIST_FILE",
    Path(__file__).with_name("whitelist.txt"),
))
LED_GPIO_PIN = int(os.environ.get("PLAKA_LED_GPIO_PIN", "17"))
FIXED_APPROVAL_GPIO_PIN = 27
LED_ON_SECONDS = float(os.environ.get("PLAKA_LED_ON_SECONDS", "5"))
CAMERA_CONTROLS = {
    "FrameRate": float(os.environ.get("PLAKA_CAMERA_FPS", "15.0")),
    "AeEnable": True,
    "AwbEnable": True,
}
failure_log_index = 0


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


class GateLed:
    def __init__(self, gpio_pin):
        self.gpio_pin = gpio_pin
        self.fixed_output_gpio_pin = FIXED_APPROVAL_GPIO_PIN
        self.gpio = None
        try:
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.fixed_output_gpio_pin, GPIO.OUT, initial=GPIO.HIGH)
            self.gpio = GPIO
        except Exception:
            pass

    def approve(self):
        if self.gpio is None:
            return
        self.gpio.output(self.gpio_pin, self.gpio.HIGH)
        time.sleep(LED_ON_SECONDS)
        self.gpio.output(self.gpio_pin, self.gpio.LOW)

    def close(self):
        if self.gpio is None:
            return
        try:
            self.gpio.output(self.gpio_pin, self.gpio.LOW)
            self.gpio.cleanup(self.gpio_pin)
            self.gpio.cleanup(self.fixed_output_gpio_pin)
        except Exception:
            pass


def main():
    imgprocess.prepare_debug_output_dir()
    sources = {}
    source_failures = {}
    last_usb_scan_at = 0.0
    gate_led = GateLed(LED_GPIO_PIN)

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
                wait_for_next_frame(frame_started_at)
                continue

            try:
                source_results = process_sources(sources.values())
                for source_name, results in source_results:
                    for result in results:
                        handle_result(source_name, result, source_failures, gate_led)
            except Exception:
                pass

            remove_failed_usb_sources(sources)
            if not sources:
                sources = start_picamera_fallback()
            wait_for_next_frame(frame_started_at)
    finally:
        close_sources(sources.values())
        gate_led.close()


def start_available_cameras():
    usb_sources = discover_usb_sources()
    if usb_sources:
        return usb_sources
    return start_picamera_fallback()


def start_picamera_fallback():
    try:
        source = start_picamera()
    except Exception:
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
    log_camera_started(
        source.name,
        actual_width,
        actual_height,
        backend="V4L2",
        fps=capture.get(cv2.CAP_PROP_FPS),
    )
    return source


def should_rescan_usb(sources, last_usb_scan_at):
    return time.monotonic() - last_usb_scan_at >= USB_CAMERA_RESCAN_SECONDS


def refresh_usb_sources(sources):
    if sources and not using_usb_sources(sources):
        usb_sources = discover_usb_sources()
        if not usb_sources:
            return sources
        close_sources(sources.values())
        return usb_sources

    current = discover_usb_sources(sources)
    removed_names = sorted(set(sources) - set(current))
    for name in removed_names:
        sources[name].close()
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
            except CameraReadError:
                source.read_failures += 1
            except Exception:
                pass
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
    controls = configuration.get("controls", {})
    fps = controls.get("FrameRate", CAMERA_CONTROLS["FrameRate"])
    log_camera_started(
        "picamera",
        width,
        height,
        backend="Picamera2",
        format=pixel_format,
        fps=fps,
    )


def log_camera_started(name, width, height, **settings):
    parts = [f"KAMERA: {name}", f"{width}x{height}"]
    for key, value in settings.items():
        if value is not None:
            parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


def apply_optional_camera_controls(camera):
    for controls in (
        {"AfMode": 2},
        {"AfSpeed": 1},
    ):
        try:
            camera.set_controls(controls)
        except Exception:
            pass


def handle_result(source_name, result, source_failures, gate_led):
    if is_failure(result) and has_debug_snapshot(result):
        source_failures[source_name] = source_failures.get(source_name, 0) + 1
        log_failure_count(source_name, source_failures[source_name])
    elif is_failure(result):
        return
    else:
        source_failures[source_name] = 0
        handle_plate_read(source_name, result, gate_led)


def is_failure(result):
    return result and not result.startswith(SUCCESS_RESULT_PREFIX)


def has_debug_snapshot(result):
    return " | debug=" in result


def log_failure_count(_source_name, failure_count):
    global failure_log_index

    failure_log_index = failure_count
    print(f"BD: {failure_log_index}", flush=True)


def handle_plate_read(_source_name, result, gate_led):
    plate = get_plate_from_success_result(result)
    if not plate:
        return

    if is_plate_whitelisted(plate):
        print(f"++ {plate}", flush=True)
        gate_led.approve()
        return

    save.write(plate, print_to_terminal=False)
    print(f"-- {plate}", flush=True)


def get_plate_from_success_result(result):
    if not result or not result.startswith(SUCCESS_RESULT_PREFIX):
        return None
    return result[len(SUCCESS_RESULT_PREFIX):].split(" | ", 1)[0]


def is_plate_whitelisted(plate):
    return plate in load_whitelist()


def load_whitelist():
    if not WHITELIST_FILE.exists():
        return set()

    plates = set()
    with open(WHITELIST_FILE, "r", encoding="utf-8") as whitelist_file:
        for line in whitelist_file:
            plate = line.strip().upper().replace(" ", "")
            if not plate or plate.startswith("#"):
                continue
            plates.add(plate)
    return plates


def wait_for_next_frame(frame_started_at):
    elapsed = time.monotonic() - frame_started_at
    remaining = FRAME_CAPTURE_INTERVAL_SECONDS - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    main()
