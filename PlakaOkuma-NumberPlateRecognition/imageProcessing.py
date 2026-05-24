import cv2
import json
import os
import save
import shutil
import re
import time
from pathlib import Path

PLATE_PATTERN = re.compile(r"^\d{2}[A-Z]{1,3}\d{2,4}$")
PREFIXED_PLATE_PATTERN = re.compile(r"^[A-Z]+(\d{2}[A-Z]{1,3}\d{2,4})$")
MIN_OCR_WIDTH = 760
MAX_OCR_WIDTH = 1000
CROP_MARGIN_RATIO = 0.12
MIN_PLATE_ASPECT_RATIO = float(os.environ.get("PLAKA_MIN_ASPECT", "2.0"))
MAX_PLATE_ASPECT_RATIO = float(os.environ.get("PLAKA_MAX_ASPECT", "6.5"))
MIN_PLATE_AREA_RATIO = float(os.environ.get("PLAKA_MIN_AREA", "0.003"))
MAX_PLATE_AREA_RATIO = float(os.environ.get("PLAKA_MAX_AREA", "0.08"))
YOLO_MODEL_PATH = Path(os.environ.get(
    "PLAKA_YOLO_MODEL",
    Path(__file__).with_name("models") / "license_plate_detector.pt",
))
YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("PLAKA_YOLO_CONF", "0.25"))
DEBUG_SAVE_ENABLED = os.environ.get("PLAKA_DEBUG", "1").lower() not in ("0", "false", "no")
DEBUG_SAVE_EVERY_SECONDS = float(os.environ.get("PLAKA_DEBUG_INTERVAL", "2"))
DEBUG_OUTPUT_DIR = Path(os.environ.get("PLAKA_DEBUG_DIR", "debug_frames"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATPLOTLIB_CACHE_DIR = Path(os.environ.get("MPLCONFIGDIR", PROJECT_ROOT / ".matplotlib_cache"))
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PADDLE_CACHE_DIR = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", PROJECT_ROOT / ".paddlex_cache"))
PADDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PADDLE_OCR_CACHE_DIR = Path(os.environ.get("PADDLE_OCR_BASE_DIR", PROJECT_ROOT / ".paddleocr_cache"))
PADDLE_OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE_DIR))
os.environ.setdefault("PADDLE_OCR_BASE_DIR", str(PADDLE_OCR_CACHE_DIR))

_last_debug_save_at = 0.0
_debug_counter = 0
_plate_detector = None
_paddle_ocr = None


def rec(img):
    frame = img.copy()
    debug_started_at = time.time()

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_license_plates(frame)
    except Exception as exc:
        result = f"yolo hata: {exc}"
        debug_dir = save_debug_snapshot(
            frame,
            None,
            [],
            None,
            None,
            result,
            debug_started_at,
            error=traceback_text(exc),
        )
        return add_debug_path(result, debug_dir)

    plate_detections = get_plate_detections_for_ocr(detections, frame.shape)
    if not plate_detections:
        result = "yolo gecerli plaka adayi bulamadi"
        debug_dir = save_debug_snapshot(
            frame,
            None,
            detections,
            None,
            None,
            result,
            debug_started_at,
        )
        return add_debug_path(result, debug_dir)

    try:
        best_read = read_best_plate_from_detections(gray, plate_detections)
    except Exception as exc:
        best_detection = select_best_detection(plate_detections)
        cropped_gray = crop_plate(gray, best_detection["bbox"]) if best_detection else None
        result = f"ocr hata: {exc}"
        debug_dir = save_debug_snapshot(
            frame,
            best_detection,
            detections,
            cropped_gray,
            cropped_gray,
            result,
            debug_started_at,
            error=traceback_text(exc),
        )
        return add_debug_path(result, debug_dir)

    if best_read is None:
        result = "plaka kirpma bos"
        debug_dir = save_debug_snapshot(
            frame,
            None,
            detections,
            None,
            None,
            result,
            debug_started_at,
        )
        return add_debug_path(result, debug_dir)

    best_detection = best_read["detection"]
    cropped_gray = best_read["cropped_gray"]
    raw_text = best_read["raw_text"]
    text = best_read["cleaned_text"]
    normalized_text = best_read["normalized_text"]
    cropped_ocr = best_read["cropped_ocr"]
    ocr_attempts = best_read["ocr_attempts"]

    if not text:
        result = "ocr metin okuyamadi"
        debug_dir = save_debug_snapshot(
            frame,
            best_detection,
            detections,
            cropped_gray,
            cropped_ocr,
            result,
            debug_started_at,
            raw_text,
            text,
            ocr_attempts=ocr_attempts,
        )
        return add_debug_path(result, debug_dir)

    if not normalized_text:
        result = f"ocr plaka formatini reddetti: ham='{raw_text}' temiz='{text}'"
        debug_dir = save_debug_snapshot(
            frame,
            best_detection,
            detections,
            cropped_gray,
            cropped_ocr,
            result,
            debug_started_at,
            raw_text,
            text,
            ocr_attempts=ocr_attempts,
        )
        return add_debug_path(result, debug_dir)
    text = normalized_text

    save.write(text, print_to_terminal=False)
    result = f"plaka kaydedildi: {text}"

    debug_dir = save_debug_snapshot(
        frame,
        best_detection,
        detections,
        cropped_gray,
        cropped_ocr,
        result,
        debug_started_at,
        raw_text,
        text,
        normalized_text,
        ocr_attempts=ocr_attempts,
        force=True,
    )
    return add_debug_path(result, debug_dir)


def prepare_debug_output_dir():
    global _last_debug_save_at, _debug_counter

    if not DEBUG_SAVE_ENABLED:
        return

    if is_unsafe_debug_output_dir(DEBUG_OUTPUT_DIR):
        raise ValueError(f"Debug klasoru temizlemek icin guvensiz: {DEBUG_OUTPUT_DIR}")

    if DEBUG_OUTPUT_DIR.exists():
        shutil.rmtree(DEBUG_OUTPUT_DIR)
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _last_debug_save_at = 0.0
    _debug_counter = 0


def is_unsafe_debug_output_dir(output_dir):
    resolved_output_dir = output_dir.resolve()
    project_dir = Path(__file__).resolve().parent
    protected_dirs = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        project_dir,
        project_dir.parent,
    }
    return resolved_output_dir in protected_dirs


def get_plate_detector():
    global _plate_detector

    if _plate_detector is None:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model dosyasi bulunamadi: {YOLO_MODEL_PATH}")
        from ultralytics import YOLO

        _plate_detector = YOLO(str(YOLO_MODEL_PATH))
    return _plate_detector


def get_paddle_ocr():
    global _paddle_ocr

    if _paddle_ocr is None:
        from paddleocr import PaddleOCR

        try:
            from paddleocr import TextRecognition
        except ImportError:
            TextRecognition = None

        init_options = []
        if TextRecognition is not None:
            init_options.append((TextRecognition, {"model_name": "en_PP-OCRv5_mobile_rec"}))
        init_options.extend((
            (
                PaddleOCR,
                {
                    "lang": "en",
                    "text_detection_model_name": "PP-OCRv5_mobile_det",
                    "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "show_log": False,
                },
            ),
            (PaddleOCR, {"lang": "en", "use_angle_cls": True, "show_log": False}),
            (PaddleOCR, {"lang": "en"}),
        ))
        errors = []
        for ocr_class, options in init_options:
            try:
                _paddle_ocr = ocr_class(**options)
                break
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        if _paddle_ocr is None:
            raise RuntimeError(f"PaddleOCR baslatilamadi: {' | '.join(errors)}")
    return _paddle_ocr


def detect_license_plates(frame):
    model = get_plate_detector()
    result = model.predict(
        frame,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]
    detections = []
    frame_height, frame_width = frame.shape[:2]

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0]) if box.cls is not None else None
        bbox = clip_bbox((x1, y1, x2, y2), frame_width, frame_height)
        if bbox is None:
            continue
        detections.append({
            "bbox": bbox,
            "confidence": confidence,
            "class_id": class_id,
        })

    return detections


def clip_bbox(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame_width - 1, int(round(x1))))
    y1 = max(0, min(frame_height - 1, int(round(y1))))
    x2 = max(0, min(frame_width - 1, int(round(x2))))
    y2 = max(0, min(frame_height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def select_best_detection(detections):
    if not detections:
        return None
    return max(detections, key=lambda detection: detection["confidence"])


def filter_plate_detections(detections, frame_shape):
    filtered = []
    for detection in detections:
        metadata = detection_metadata(detection, frame_shape)
        if metadata is None:
            continue
        aspect_ratio = metadata["aspect_ratio"]
        area_ratio = metadata["area_ratio"]
        detection["aspect_ratio"] = aspect_ratio
        detection["area_ratio"] = area_ratio
        if (
            MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO
            and MIN_PLATE_AREA_RATIO <= area_ratio <= MAX_PLATE_AREA_RATIO
        ):
            filtered.append(detection)

    return sorted(filtered, key=detection_priority, reverse=True)


def get_plate_detections_for_ocr(detections, frame_shape):
    filtered = filter_plate_detections(detections, frame_shape)
    if filtered:
        return filtered

    # The YOLO model is already specialized for plates; if local geometry
    # thresholds reject every box, keep the best YOLO boxes for OCR/debugging.
    for detection in detections:
        metadata = detection_metadata(detection, frame_shape)
        if metadata is None:
            continue
        detection["aspect_ratio"] = metadata["aspect_ratio"]
        detection["area_ratio"] = metadata["area_ratio"]
        detection["geometry_filter_rejected"] = True

    return sorted(detections, key=lambda detection: detection["confidence"], reverse=True)


def detection_priority(detection):
    aspect_ratio = detection.get("aspect_ratio", 0)
    aspect_penalty = abs(aspect_ratio - 4.0) / 4.0
    return detection["confidence"] - (aspect_penalty * 0.15)


def read_best_plate_from_detections(gray, detections):
    fallback = None

    for index, detection in enumerate(detections):
        cropped_gray = crop_plate(gray, detection["bbox"])
        if cropped_gray.size == 0:
            continue

        ocr_candidates = build_ocr_candidates(cropped_gray)
        raw_text, text, cropped_ocr, ocr_attempts = read_plate_text(ocr_candidates)
        for attempt in ocr_attempts:
            attempt["detection_index"] = index
            attempt["detection_confidence"] = detection["confidence"]

        normalized_text = normalize_plate_text(text)
        read = {
            "detection": detection,
            "cropped_gray": cropped_gray,
            "raw_text": raw_text,
            "cleaned_text": text,
            "normalized_text": normalized_text,
            "cropped_ocr": cropped_ocr,
            "ocr_attempts": ocr_attempts,
        }
        if normalized_text:
            return read
        if fallback is None or ocr_fallback_score(read) > ocr_fallback_score(fallback):
            fallback = read

    return fallback


def ocr_fallback_score(read):
    text = read["cleaned_text"] or ""
    detection = read["detection"]
    length_score = min(len(text), 8) / 8.0
    aspect_ratio = detection.get("aspect_ratio", 0)
    aspect_score = max(0, 1 - abs(aspect_ratio - 4.0) / 4.0)
    return length_score + detection["confidence"] + (aspect_score * 0.25)


def crop_plate(gray, bbox):
    x1, y1, x2, y2 = bbox
    height = y2 - y1 + 1
    width = x2 - x1 + 1
    margin_x = int(height * CROP_MARGIN_RATIO)
    margin_y = int(width * CROP_MARGIN_RATIO)
    top = max(0, y1 - margin_x)
    left = max(0, x1 - margin_y)
    bottom = min(gray.shape[0] - 1, y2 + margin_x)
    right = min(gray.shape[1] - 1, x2 + margin_y)
    return gray[top:bottom + 1, left:right + 1]


def build_ocr_candidates(cropped):
    trimmed = trim_plate_crop(cropped)
    deskewed_full = deskew_plate(cropped)
    deskewed_trimmed = deskew_plate(trimmed)

    candidates = []
    for source_name, source in (
        ("full", cropped),
        ("deskewed_full", deskewed_full),
        ("trimmed", trimmed),
        ("deskewed_trimmed", deskewed_trimmed),
    ):
        if source is None or source.size == 0:
            continue
        resized = resize_for_ocr(source)
        candidates.extend(prepare_for_paddleocr(resized, source_name))
    return candidates


def trim_plate_crop(cropped):
    height, width = cropped.shape[:2]
    if height < 20 or width < 60:
        return cropped

    top = int(height * 0.10)
    bottom = int(height * 0.92)
    left = int(width * 0.01)
    right = int(width * 0.99)
    trimmed = cropped[top:bottom, left:right]
    return trimmed if trimmed.size > 0 else cropped


def deskew_plate(cropped):
    threshold = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    merged = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = select_plate_like_contour(contours, cropped.shape)
    if contour is None:
        return cropped

    angle = cv2.minAreaRect(contour)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 1 or abs(angle) > 20:
        return cropped

    height, width = cropped.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        cropped,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def select_plate_like_contour(contours, image_shape):
    height, width = image_shape[:2]
    image_area = height * width
    best = None
    best_area = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect_ratio = w / float(h)
        area = w * h
        if 1.8 <= aspect_ratio <= 7.0 and area >= image_area * 0.08 and area > best_area:
            best = contour
            best_area = area

    return best


def resize_for_ocr(cropped):
    height, width = cropped.shape[:2]
    if width < MIN_OCR_WIDTH:
        scale = MIN_OCR_WIDTH / float(width)
    elif width > MAX_OCR_WIDTH:
        scale = MAX_OCR_WIDTH / float(width)
    else:
        return cropped
    resized_height = max(1, int(height * scale))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(cropped, (int(width * scale), resized_height), interpolation=interpolation)


def prepare_for_paddleocr(cropped, source_name):
    contrast = enhance_contrast(cropped)
    sharpened = sharpen_image(contrast)
    denoised = cv2.bilateralFilter(sharpened, 7, 45, 45)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return (
        (f"{source_name}_gray", cropped),
        (f"{source_name}_contrast", contrast),
        (f"{source_name}_sharpened", sharpened),
        (f"{source_name}_otsu", otsu),
        (f"{source_name}_adaptive", adaptive),
    )


def enhance_contrast(cropped):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(cropped)


def sharpen_image(cropped):
    blurred = cv2.GaussianBlur(cropped, (0, 0), 1.2)
    return cv2.addWeighted(cropped, 1.7, blurred, -0.7, 0)


def read_plate_text(candidates):
    attempts = []
    fallback = ("", "", candidates[0][1] if candidates else None)
    ocr = get_paddle_ocr()

    for candidate_name, candidate_image in candidates:
        raw_text, confidence = run_paddle_ocr(ocr, candidate_image)
        cleaned_text = clean_ocr_text(raw_text)
        attempts.append({
            "candidate": candidate_name,
            "raw": raw_text,
            "cleaned": cleaned_text,
            "confidence": confidence,
        })

        if cleaned_text and not fallback[1]:
            fallback = (raw_text, cleaned_text, candidate_image)
        if normalize_plate_text(cleaned_text):
            return raw_text, cleaned_text, candidate_image, attempts

    raw_text, cleaned_text, candidate_image = fallback
    return raw_text, cleaned_text, candidate_image, attempts


def run_paddle_ocr(ocr, candidate_image):
    image = ensure_bgr(candidate_image)
    errors = []

    if hasattr(ocr, "ocr"):
        for kwargs in (
            {"det": False, "rec": True, "cls": False},
            {"det": False, "rec": True},
            {},
        ):
            try:
                result = ocr.ocr(image, **kwargs)
                return best_ocr_text(result)
            except (TypeError, ValueError, RuntimeError) as exc:
                errors.append(str(exc))

    if hasattr(ocr, "predict"):
        try:
            result = ocr.predict(image)
            return best_ocr_text(result)
        except (TypeError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))

    raise RuntimeError(f"PaddleOCR cagrisi basarisiz: {' | '.join(errors)}")


def ensure_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def best_ocr_text(result):
    texts = extract_ocr_texts(result)
    if not texts:
        return "", None

    texts.sort(key=lambda item: item[1] if item[1] is not None else 0, reverse=True)
    raw_text = "".join(text for text, _ in texts)
    best_confidence = texts[0][1]
    return raw_text.strip(), best_confidence


def extract_ocr_texts(value):
    if value is None:
        return []

    if isinstance(value, dict):
        if "rec_texts" in value:
            scores = value.get("rec_scores") or []
            return [
                (str(text), score_at(scores, index))
                for index, text in enumerate(value.get("rec_texts") or [])
                if text
            ]
        for text_key in ("text", "rec_text"):
            if text_key in value and value[text_key]:
                return [(str(value[text_key]), value.get("score") or value.get("rec_score"))]
        texts = []
        for item in value.values():
            texts.extend(extract_ocr_texts(item))
        return texts

    if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], str):
        return [(value[0], value[1] if isinstance(value[1], (int, float)) else None)]

    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], str):
            return [(value[0], value[1] if isinstance(value[1], (int, float)) else None)]

        texts = []
        for item in value:
            texts.extend(extract_ocr_texts(item))
        return texts

    return []


def score_at(scores, index):
    if index < len(scores):
        score = scores[index]
        return score if isinstance(score, (int, float)) else None
    return None


def clean_ocr_text(raw_text):
    cleaned_text = raw_text.replace(" ", "").upper()
    cleaned_text = re.sub(r"[^0-9A-Z]", "", cleaned_text)
    return cleaned_text


def normalize_plate_text(text):
    if PLATE_PATTERN.match(text):
        return text if is_valid_city_code(text[:2]) else None

    match = PREFIXED_PLATE_PATTERN.match(text)
    if match:
        plate = match.group(1)
        return plate if is_valid_city_code(plate[:2]) else None

    return None


def is_valid_city_code(city_code):
    return 1 <= int(city_code) <= 81


def save_debug_snapshot(
    frame,
    plate_detection,
    detections,
    cropped_gray,
    cropped_ocr,
    result,
    started_at,
    raw_text=None,
    cleaned_text=None,
    normalized_text=None,
    ocr_attempts=None,
    error=None,
    force=False,
):
    if not should_save_debug_snapshot(force):
        return None

    debug_dir = make_debug_dir(started_at)
    annotated = frame.copy()
    draw_debug_detections(annotated, detections, plate_detection)

    cv2.imwrite(str(debug_dir / "01_frame_raw.jpg"), frame)
    cv2.imwrite(str(debug_dir / "03_frame_annotated.jpg"), annotated)
    if cropped_gray is not None and cropped_gray.size > 0:
        cv2.imwrite(str(debug_dir / "04_crop_gray.jpg"), cropped_gray)
    if cropped_ocr is not None and cropped_ocr.size > 0:
        cv2.imwrite(str(debug_dir / "05_crop_ocr_input.jpg"), cropped_ocr)

    metadata = {
        "result": result,
        "raw_ocr": raw_text,
        "cleaned_ocr": cleaned_text,
        "normalized_plate": normalized_text,
        "ocr_attempts": ocr_attempts or [],
        "selected_detection": detection_metadata(plate_detection, frame.shape),
        "candidate_count": len(detections),
        "detections": [
            detection_metadata(detection, frame.shape)
            for detection in detections
        ],
        "error": error,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
    }
    with open(debug_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    return debug_dir


def should_save_debug_snapshot(force=False):
    global _last_debug_save_at

    if not DEBUG_SAVE_ENABLED:
        return False
    now = time.monotonic()
    if force or now - _last_debug_save_at >= DEBUG_SAVE_EVERY_SECONDS:
        _last_debug_save_at = now
        return True
    return False


def make_debug_dir(started_at):
    global _debug_counter

    _debug_counter += 1
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    debug_dir = DEBUG_OUTPUT_DIR / f"{timestamp}_{_debug_counter:04d}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def draw_debug_detections(annotated, detections, selected_detection):
    selected_bbox = selected_detection["bbox"] if selected_detection else None
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        color = (0, 255, 0) if detection["bbox"] == selected_bbox else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated,
            f"{detection['confidence']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def detection_metadata(detection, frame_shape):
    if detection is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    image_area = frame_height * frame_width
    x1, y1, x2, y2 = detection["bbox"]
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    return {
        "x": int(x1),
        "y": int(y1),
        "width": int(width),
        "height": int(height),
        "confidence": detection["confidence"],
        "class_id": detection["class_id"],
        "aspect_ratio": width / float(height) if height else None,
        "area_ratio": (width * height) / image_area if image_area else None,
    }


def add_debug_path(result, debug_dir):
    if debug_dir is None:
        return result
    return f"{result} | debug={debug_dir}"


def traceback_text(exc):
    return f"{type(exc).__name__}: {exc}"
