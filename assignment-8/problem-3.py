import sounddevice as sd
import soundfile as sf
from pathlib import Path
import scipy.io
import scipy.signal as sig
import pysptk.sptk
import numpy as np


def transform_vowel(source, target, sample_rate: int):
    order = 10
    a_src = pysptk.sptk.lpc(source, order=order)
    a_trg = pysptk.sptk.lpc(target, order=order)

    a_src = np.concatenate(([1.0], -a_src[1:] / a_src[0]))
    a_trg = np.concatenate(([1.0], -a_trg[1:] / a_trg[0]))

    residual = sig.lfilter(a_src, [1.0], source)

    out = sig.lfilter([1.0], a_trg, residual)
    print(out)

    sf.write("debug_out.wav", out, sample_rate)
    sd.play(out, sample_rate)
    sd.wait()


def main():
    file = Path("vowels.mat")
    mat = scipy.io.loadmat(file)
    sample_rate = int(mat["fs"][0][0])
    vowels = mat["v"][0]

    a = vowels[0].squeeze()
    e = vowels[1].squeeze()
    i = vowels[2].squeeze()

    my_aaa, sr = sf.read("aaa.wav")

    if sr != sample_rate:
        raise RuntimeError("Sample rate mismatch")

    # recording = sd.rec(int(5 * sample_rate), samplerate=sample_rate, channels=1)
    # sd.wait()
    # sd.play(recording, sample_rate)
    # sd.wait()

    transform_vowel(my_aaa, i, sample_rate)


if __name__ == "__main__":
    main()
