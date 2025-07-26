def load_audio(file_path, sample_rate=16000):
    import torchaudio

    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform

def save_audio(file_path, waveform, sample_rate=16000):
    import torchaudio

    torchaudio.save(file_path, waveform, sample_rate)

def normalize_audio(waveform):
    return waveform / torch.max(torch.abs(waveform))

def trim_audio(waveform, start_time, end_time, sample_rate=16000):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return waveform[:, start_sample:end_sample]

def pad_audio(waveform, target_length):
    import torch.nn.functional as F

    current_length = waveform.size(1)
    if current_length < target_length:
        padding = target_length - current_length
        waveform = F.pad(waveform, (0, padding), mode='constant', value=0)
    return waveform

def get_audio_duration(waveform, sample_rate=16000):
    return waveform.size(1) / sample_rate