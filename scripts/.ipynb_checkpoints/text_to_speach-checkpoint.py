import torch
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
from torch import no_grad, LongTensor
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def synthesize_audio(model_path, config_path, text_to_read, speaker_id, cleaned=False, length_scale=1.0, noise_scale=0.667, noise_scale_w=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hps_ms = utils.get_hparams_from_file(config_path)
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model).to(device)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model_path, net_g_ms)

    stn_tst = get_text(text_to_read, hps_ms, cleaned=cleaned)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
    return audio