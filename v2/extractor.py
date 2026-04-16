# extractor.py
import logging
import numpy as np
import librosa
from typing import List, Tuple

from config import SR, HOP_LENGTH, N_MFCC, SILENCE_DB, FEATURE_DIM, FMIN, FMAX, TOP_DB

log = logging.getLogger(__name__)


def extract_features(audio: np.ndarray, sr: int) -> List[float]:
    """
    Tính vector 18 chiều từ một đoạn âm thanh (đã được cắt).
    Giống như _extract_worker cũ nhưng áp dụng cho segment.
    """
    # Time-domain
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    avg_energy = float(np.mean(rms))

    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    avg_zcr = float(np.mean(zcr))

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    silence_ratio = float(np.sum(rms_db < SILENCE_DB) / rms.shape[1]) if rms.shape[1] > 0 else 0.0

    # Frequency-domain
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=FMIN, fmax=FMAX, sr=sr, hop_length=HOP_LENGTH
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    avg_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)
    avg_centroid = float(np.mean(spec_cent))

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    avg_mfccs = np.mean(mfccs, axis=1).tolist()

    return [avg_energy, avg_zcr, silence_ratio, avg_pitch, avg_centroid] + avg_mfccs


def segment_audio(file_path: str) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    Đọc file âm thanh, phát hiện khoảng lặng và trả về:
      - segments: list các mảng numpy (mỗi mảng là tín hiệu âm thanh của một đoạn)
      - intervals: list (start_time, end_time) tính bằng giây
    """
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    # Dùng split của librosa để tách theo khoảng lặng
    intervals = librosa.effects.split(y, top_db=TOP_DB)
    segments = [y[start:end] for start, end in intervals]
    # Chuyển đổi sample index sang giây
    time_intervals = [(start / sr, end / sr) for start, end in intervals]
    return segments, time_intervals


def process_file(file_path: str) -> List[Tuple[np.ndarray, float, float, List[float]]]:
    """
    Xử lý một file: phân đoạn và trích xuất vector cho mỗi đoạn.
    Trả về list các tuple: (audio_segment, start_time, end_time, feature_vector)
    """
    try:
        segments, intervals = segment_audio(file_path)
        results = []
        for seg, (start, end) in zip(segments, intervals):
            if len(seg) < HOP_LENGTH:  # bỏ qua đoạn quá ngắn
                continue
            vec = extract_features(seg, SR)
            results.append((seg, start, end, vec))
        return results
    except Exception as e:
        log.error("Error processing %s: %s", file_path, e)
        return []