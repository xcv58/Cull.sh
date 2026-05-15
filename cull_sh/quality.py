from __future__ import annotations

import warnings

import numpy as np

from cull_sh.models import LocalQualityMetrics

_IQA_METRICS: dict[str, object] | None = None
_IQA_IMPORT_ERROR: str | None = None
_SUPPORT_METRICS: dict[str, object] | None = None
_SUPPORT_IMPORT_ERRORS: dict[str, str] = {}


def analyze_local_quality(
    image_bytes: bytes,
    include_portrait_metrics: bool = False,
    enable_learned_iqa: bool = True,
    enable_brisque: bool = True,
    enable_cpbd: bool = True,
) -> LocalQualityMetrics:
    """
    Compute fast local quality metrics from preview JPEG bytes.

    The current implementation favors cheap, parallel-safe metrics:
    - Laplacian variance for sharpness
    - Tenengrad for edge strength
    - brightness / contrast heuristics
    - optional portrait hints via OpenCV Haar cascades
    """
    import cv2

    cv2.setNumThreads(1)

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode preview bytes into an image")
    grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    metrics = LocalQualityMetrics(
        blur_score=_laplacian_variance(grayscale),
        tenengrad_score=_tenengrad_score(grayscale),
        brightness_mean=float(grayscale.mean()),
        contrast_stddev=float(grayscale.std()),
        perceptual_hash=_difference_hash(grayscale),
    )

    if enable_learned_iqa:
        musiq_score, nima_score = _learned_iqa_scores(bgr)
        metrics.musiq_score = musiq_score
        metrics.nima_score = nima_score
    if enable_brisque:
        metrics.brisque_score = _brisque_score(bgr)
    if enable_cpbd:
        metrics.cpbd_score = _cpbd_score(grayscale)

    if include_portrait_metrics:
        face_count, eye_count = _detect_faces_and_eyes(grayscale)
        metrics.face_count = face_count
        metrics.eye_count = eye_count

    metrics.local_rank_score = compute_local_rank_score(metrics, portrait_mode=include_portrait_metrics)
    return metrics


def should_reject_for_local_quality(
    metrics: LocalQualityMetrics,
    min_blur_score: float,
    min_tenengrad_score: float,
    min_musiq_score: float,
    min_nima_score: float,
    max_brisque_score: float,
    min_cpbd_score: float,
    use_brisque_for_reject: bool,
    use_cpbd_for_reject: bool,
    local_reject_required_support_votes: int,
) -> bool:
    trace = build_local_decision_trace(
        metrics,
        min_blur_score=min_blur_score,
        min_tenengrad_score=min_tenengrad_score,
        min_musiq_score=min_musiq_score,
        min_nima_score=min_nima_score,
        max_brisque_score=max_brisque_score,
        min_cpbd_score=min_cpbd_score,
        use_brisque_for_reject=use_brisque_for_reject,
        use_cpbd_for_reject=use_cpbd_for_reject,
        local_reject_required_support_votes=local_reject_required_support_votes,
    )
    return bool(trace["quality_reject"])


def compute_local_rank_score(
    metrics: LocalQualityMetrics,
    portrait_mode: bool = False,
) -> float:
    blur_score = metrics.blur_score or 0.0
    tenengrad_score = metrics.tenengrad_score or 0.0
    musiq_score = metrics.musiq_score or 0.0
    nima_score = metrics.nima_score or 0.0
    cpbd_score = metrics.cpbd_score or 0.0
    brisque_score = metrics.brisque_score or 0.0
    contrast_stddev = metrics.contrast_stddev or 0.0
    brightness_mean = metrics.brightness_mean or 118.0

    score = blur_score + (tenengrad_score * 4.0) + (contrast_stddev * 0.5)
    score += musiq_score * 3.0
    score += nima_score * 25.0
    score += cpbd_score * 60.0
    score -= brisque_score * 0.8
    score -= abs(brightness_mean - 118.0) * 0.15

    if portrait_mode:
        score += float((metrics.face_count or 0) * 20)
        score += float((metrics.eye_count or 0) * 10)

    return float(score)


def _laplacian_variance(grayscale: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def _tenengrad_score(grayscale: np.ndarray) -> float:
    import cv2

    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y))
    return float(gradient_magnitude.mean())


def _detect_faces_and_eyes(grayscale: np.ndarray) -> tuple[int | None, int | None]:
    import cv2

    face_cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    eye_cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_eye.xml"

    face_classifier = cv2.CascadeClassifier(face_cascade_path)
    eye_classifier = cv2.CascadeClassifier(eye_cascade_path)
    if face_classifier.empty() or eye_classifier.empty():
        return None, None

    faces = face_classifier.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40),
    )
    eye_count = 0
    for (x, y, width, height) in faces:
        region = grayscale[y : y + height, x : x + width]
        eyes = eye_classifier.detectMultiScale(
            region,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(12, 12),
        )
        eye_count += len(eyes)

    return int(len(faces)), int(eye_count)


def _difference_hash(grayscale: np.ndarray) -> str:
    import cv2

    resized = cv2.resize(grayscale, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    packed = 0
    for bit in diff.flatten():
        packed = (packed << 1) | int(bit)
    return f"{packed:016x}"


def learned_iqa_available() -> bool:
    if _IQA_METRICS is not None:
        return True
    _load_iqa_metrics()
    return _IQA_METRICS is not None


def learned_iqa_import_error() -> str | None:
    if _IQA_METRICS is None and _IQA_IMPORT_ERROR is None:
        _load_iqa_metrics()
    return _IQA_IMPORT_ERROR


def support_metric_import_errors() -> dict[str, str]:
    if _SUPPORT_METRICS is None and not _SUPPORT_IMPORT_ERRORS:
        _load_support_metrics()
    return dict(_SUPPORT_IMPORT_ERRORS)


def _learned_iqa_scores(bgr: np.ndarray) -> tuple[float | None, float | None]:
    scorers = _load_iqa_metrics()
    if scorers is None:
        return None, None

    import cv2
    import torch

    torch.set_num_threads(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    musiq_metric = scorers["musiq"]
    nima_metric = scorers["nima"]
    with torch.inference_mode():
        musiq_score = float(musiq_metric(image_tensor).detach().cpu().item())
        nima_score = float(nima_metric(image_tensor).detach().cpu().item())
    return musiq_score, nima_score


def build_local_decision_trace(
    metrics: LocalQualityMetrics,
    *,
    min_blur_score: float,
    min_tenengrad_score: float,
    min_musiq_score: float,
    min_nima_score: float,
    max_brisque_score: float,
    min_cpbd_score: float,
    use_brisque_for_reject: bool,
    use_cpbd_for_reject: bool,
    local_reject_required_support_votes: int,
) -> dict[str, object]:
    votes = {
        "laplacian_weak": (metrics.blur_score or 0.0) < min_blur_score,
        "tenengrad_weak": (metrics.tenengrad_score or 0.0) < min_tenengrad_score,
        "musiq_weak": None if metrics.musiq_score is None else metrics.musiq_score < min_musiq_score,
        "nima_weak": None if metrics.nima_score is None else metrics.nima_score < min_nima_score,
        "brisque_weak": (
            None if metrics.brisque_score is None else metrics.brisque_score > max_brisque_score
        ),
        "cpbd_weak": None if metrics.cpbd_score is None else metrics.cpbd_score < min_cpbd_score,
    }
    active_support_votes = {
        "musiq": votes["musiq_weak"],
        "nima": votes["nima_weak"],
    }
    if use_brisque_for_reject:
        active_support_votes["brisque"] = votes["brisque_weak"]
    if use_cpbd_for_reject:
        active_support_votes["cpbd"] = votes["cpbd_weak"]

    available_support_votes = {
        name: value for name, value in active_support_votes.items() if value is not None
    }
    required_support_votes = (
        min(local_reject_required_support_votes, len(available_support_votes))
        if available_support_votes
        else 0
    )
    weak_support_count = sum(1 for value in available_support_votes.values() if value)
    primary_weak = bool(votes["laplacian_weak"] and votes["tenengrad_weak"])
    quality_reject = primary_weak and (
        weak_support_count >= required_support_votes if available_support_votes else True
    )

    if quality_reject:
        explanation = (
            "Rejected by local quality gate: primary sharpness weak and sufficient support signals weak."
        )
    elif primary_weak:
        explanation = (
            "Kept for review/model: primary sharpness weak but support signals were mixed or stronger."
        )
    else:
        explanation = "Passed local quality gate: primary sharpness signals were acceptable."

    return {
        "scores": {
            "laplacian": metrics.blur_score,
            "tenengrad": metrics.tenengrad_score,
            "musiq": metrics.musiq_score,
            "nima": metrics.nima_score,
            "brisque": metrics.brisque_score,
            "cpbd": metrics.cpbd_score,
            "brightness_mean": metrics.brightness_mean,
            "contrast_stddev": metrics.contrast_stddev,
            "local_rank_score": metrics.local_rank_score,
        },
        "thresholds": {
            "min_blur_score": min_blur_score,
            "min_tenengrad_score": min_tenengrad_score,
            "min_musiq_score": min_musiq_score,
            "min_nima_score": min_nima_score,
            "max_brisque_score": max_brisque_score,
            "min_cpbd_score": min_cpbd_score,
            "local_reject_required_support_votes": local_reject_required_support_votes,
        },
        "votes": votes,
        "active_support_votes": active_support_votes,
        "available_support_votes": available_support_votes,
        "required_support_votes": required_support_votes,
        "weak_support_count": weak_support_count,
        "quality_reject": quality_reject,
        "quality_explanation": explanation,
    }


def _load_iqa_metrics() -> dict[str, object] | None:
    global _IQA_METRICS
    global _IQA_IMPORT_ERROR

    if _IQA_METRICS is not None:
        return _IQA_METRICS
    if _IQA_IMPORT_ERROR is not None:
        return None

    try:
        import pyiqa
    except Exception as exc:  # pragma: no cover - optional dependency
        _IQA_IMPORT_ERROR = str(exc)
        return None

    try:
        _IQA_METRICS = {
            "musiq": pyiqa.create_metric("musiq", device="cpu"),
            "nima": pyiqa.create_metric("nima", device="cpu"),
        }
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        _IQA_IMPORT_ERROR = str(exc)
        _IQA_METRICS = None
        return None

    return _IQA_METRICS


def _load_support_metrics() -> dict[str, object] | None:
    global _SUPPORT_METRICS
    global _SUPPORT_IMPORT_ERRORS

    if _SUPPORT_METRICS is not None:
        return _SUPPORT_METRICS

    scorers: dict[str, object] = {}

    try:
        from brisque import BRISQUE

        _patch_brisque_scale_features(BRISQUE)
        scorers["brisque"] = BRISQUE()
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        _SUPPORT_IMPORT_ERRORS["brisque"] = str(exc)

    try:
        import imageio
        import scipy.ndimage

        scipy.ndimage.imread = imageio.v2.imread
        import cpbd

        scorers["cpbd"] = cpbd
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        _SUPPORT_IMPORT_ERRORS["cpbd"] = str(exc)

    _SUPPORT_METRICS = scorers
    return _SUPPORT_METRICS


def _brisque_score(bgr: np.ndarray) -> float | None:
    scorers = _load_support_metrics()
    scorer = scorers.get("brisque") if scorers is not None else None
    if scorer is None:
        return None

    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore"):
                score = float(scorer.score(rgb))
    except Exception:
        return None
    if np.isfinite(score):
        return score
    return None


def _cpbd_score(grayscale: np.ndarray) -> float | None:
    scorers = _load_support_metrics()
    scorer = scorers.get("cpbd") if scorers is not None else None
    if scorer is None:
        return None

    try:
        score = float(scorer.compute(grayscale))
    except Exception:
        return None
    if np.isfinite(score):
        return score
    return None


def _patch_brisque_scale_features(brisque_class: type) -> None:
    if getattr(brisque_class, "_cullsh_scale_patch", False):
        return

    def patched_scale_features(self, features):
        def flatten(values):
            for value in values:
                if isinstance(value, np.ndarray):
                    yield from flatten(value.flatten().tolist())
                elif isinstance(value, (list, tuple)):
                    yield from flatten(value)
                else:
                    yield float(value)

        min_flat = np.array(list(flatten(self.scale_params["min_"])), dtype=np.float64)
        max_flat = np.array(list(flatten(self.scale_params["max_"])), dtype=np.float64)
        feat_flat = np.array(list(flatten(features)), dtype=np.float64)
        return -1 + (2.0 / (max_flat - min_flat) * (feat_flat - min_flat))

    brisque_class.scale_features = patched_scale_features
    brisque_class._cullsh_scale_patch = True
