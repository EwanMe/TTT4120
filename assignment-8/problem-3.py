import sounddevice as sd
import soundfile as sf
from pathlib import Path
import scipy.io
import scipy.signal as sig
import pysptk.sptk
import numpy as np


def transform_vowel(source, target):
    order = 10
    a_src = pysptk.sptk.lpc(source / np.max(source), order=order)
    a_trg = pysptk.sptk.lpc(target / np.max(target), order=order)

    a_s_0 = a_src[0]
    a_t_0 = a_trg[0]

    a_src[0] = 1
    a_trg[0] = 1

    residual = sig.lfilter(a_src, [a_s_0], source)

    return sig.lfilter([a_t_0], a_trg, residual)


def main():
    file = Path("vowels.mat")
    mat = scipy.io.loadmat(file)
    sample_rate = int(mat["fs"][0][0])
    vowels = mat["v"][0]

    # a = vowels[0].squeeze()
    # e = vowels[1].squeeze()
    # i = vowels[2].squeeze()
    # o = vowels[3].squeeze()
    # u = vowels[4].squeeze()
    # y = vowels[5].squeeze()
    # æ = vowels[6].squeeze()
    # ø = vowels[7].squeeze()
    # å = vowels[8].squeeze()

    my_aaa, sr = sf.read("aaa.wav")

    if sr != sample_rate:
        raise RuntimeError("Sample rate mismatch")

    sd.play(my_aaa, samplerate=sample_rate)
    sd.wait()

    for i in range(9):
        vowel = vowels[i].squeeze()
        out = transform_vowel(my_aaa, vowel)

        sd.play(out, sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
