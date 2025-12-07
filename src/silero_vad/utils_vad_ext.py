from typing import Callable, List
import warnings
import numpy as np


class NumpyOnnxWrapper:
    def __init__(self, path: str, force_onnx_cpu: bool = False):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        warnings.warn('intra_op_num_threads is set to 2')
        warnings.warn('inter_op_num_threads is set to 1')

        if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        if '16k' in path:
            warnings.warn('This model support only 16000 sampling rate!')
            self.sample_rates = [16000]
        else:
            self.sample_rates = [8000, 16000]

        self._state = None
        self._context = None
        self._last_sr = 0
        self._last_batch_size = 0
        self.reset_states(1)

    def reset_states(self, batch_size: int = 1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((batch_size, 0), dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = batch_size

    def _validate_input(self, x, sr: int):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x.astype(np.float32, copy=False), sr

    def __call__(self, x, sr: int):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if (not self._last_batch_size) or (self._last_batch_size != batch_size) or (self._last_sr and self._last_sr != sr):
            self.reset_states(batch_size)

        if self._context.shape[1] == 0:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x_in = np.concatenate([self._context, x], axis=1)
        if sr in [8000, 16000]:
            ort_inputs = {'input': x_in, 'state': self._state, 'sr': np.array(sr, dtype='int64')}
            out, state = self.session.run(None, ort_inputs)
            self._state = state.astype(np.float32, copy=False)
        else:
            raise ValueError()

        self._context = x_in[:, -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        # 返回 numpy 概率 (batch, 1)
        return out

    def audio_forward(self, x, sr: int):
        x = np.asarray(x)
        x, sr = self._validate_input(x, sr)
        self.reset_states(x.shape[0])
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = np.pad(x, ((0, 0), (0, pad_num)), mode='constant', constant_values=0.0)

        outs = []
        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i + num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        return np.concatenate(outs, axis=1)


def get_speech_timestamps_np(audio,
                             model,
                             threshold: float = 0.5,
                             sampling_rate: int = 16000,
                             min_speech_duration_ms: int = 250,
                             max_speech_duration_s: float = float('inf'),
                             min_silence_duration_ms: int = 100,
                             speech_pad_ms: int = 30,
                             return_seconds: bool = False,
                             time_resolution: int = 1,
                             visualize_probs: bool = False,
                             progress_tracking_callback: Callable[[float], None] = None,
                             neg_threshold: float = None,
                             window_size_samples: int = 512,
                             min_silence_at_max_speech: float = 98,
                             use_max_poss_sil_at_max_speech: bool = True):

    """
    纯 numpy / onnxruntime 版本的语音时间戳提取，功能与 `utils_vad.get_speech_timestamps` 等价。

    仅使用 numpy 作为 onnx 模型输入与后处理，参数与返回值与原函数保持一致，
    但 `audio` 接受一维 numpy 数组或可转为一维 numpy 的序列。
    """

    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate not in [8000, 16000]:
        raise ValueError("Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates")

    window_size_samples = 512 if sampling_rate == 16000 else 256
    hop_size_samples = int(window_size_samples)
    context_size = 64 if sampling_rate == 16000 else 32

    if hasattr(model, 'session'):
        session = model.session
    elif hasattr(model, 'run'):
        session = model
    else:
        raise TypeError('model must be an onnxruntime.InferenceSession or a wrapper exposing .session')

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, context_size), dtype=np.float32)

    def infer_chunk(chunk_1d: np.ndarray, sr: int, state_arr: np.ndarray, ctx_arr: np.ndarray):
        if chunk_1d.shape[0] < window_size_samples:
            pad_len = window_size_samples - chunk_1d.shape[0]
            if pad_len > 0:
                chunk_1d = np.pad(chunk_1d, (0, pad_len), mode='constant', constant_values=0.0)
        x_b = chunk_1d.reshape(1, -1).astype(np.float32, copy=False)
        x_in = np.concatenate([ctx_arr, x_b], axis=1)
        ort_inputs = {
            'input': x_in,
            'state': state_arr,
            'sr': np.array(sr, dtype=np.int64)
        }
        out, new_state = session.run(None, ort_inputs)
        new_ctx = x_in[:, -context_size:]
        prob = float(np.asarray(out).reshape(-1)[0])
        return prob, new_state.astype(np.float32, copy=False), new_ctx

    audio_length_samples = int(audio.shape[0])
    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, hop_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        speech_prob, state, context = infer_chunk(chunk, sampling_rate, state, context)
        speech_probs.append(speech_prob)

        progress = current_start_sample + hop_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        progress_percent = (progress / audio_length_samples) * 100 if audio_length_samples else 100.0
        if progress_tracking_callback:
            progress_tracking_callback(progress_percent)

    triggered = False
    speeches = []
    current_speech = {}

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0
    prev_end = 0
    next_start = 0
    possible_ends = []

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            if temp_end != 0:
                sil_dur = (hop_size_samples * i) - temp_end
                if sil_dur > min_silence_samples_at_max_speech:
                    possible_ends.append((temp_end, sil_dur))
                temp_end = 0
            if next_start < prev_end:
                next_start = hop_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = hop_size_samples * i
            continue

        if triggered and (hop_size_samples * i) - current_speech['start'] > max_speech_samples:
            if possible_ends:
                if use_max_poss_sil_at_max_speech:
                    prev_end, dur = max(possible_ends, key=lambda x: x[1])
                else:
                    prev_end, dur = possible_ends[-1]
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur
                if next_start < prev_end + hop_size_samples * i:
                    current_speech['start'] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                current_speech['end'] = hop_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = hop_size_samples * i
            if (hop_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for speech_dict in speeches:
            speech_dict['start'] = max(round(speech_dict['start'] / sampling_rate, time_resolution), 0)
            speech_dict['end'] = min(round(speech_dict['end'] / sampling_rate, time_resolution), audio_length_seconds)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    if visualize_probs:
        # 延迟导入，避免图形依赖对纯 numpy 使用者的影响
        from .utils_vad import make_visualization
        make_visualization(speech_probs, hop_size_samples / sampling_rate)

    return speeches


