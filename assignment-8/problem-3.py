import sounddevice as sd
from pathlib import Path
import scipy.io
import scipy.signal
import pysptk.sptk
import numpy as np


def transform_vowel(vowel_1, vowel_2, sample_rate: int):
    a_1 = pysptk.sptk.lpc(vowel_1, order=10)
    a_1 /= a_1[0]
    residual = scipy.signal.lfilter(a_1[0], [1.0], vowel_1)

    a_2 = pysptk.sptk.lpc(vowel_2, order=10)
    a_2 /= a_2[0]
    transformed = scipy.signal.lfilter([1.0], a_2, residual)

    sd.play(transformed, sample_rate)
    sd.wait()


def main():
    file = Path("vowels.mat")
    mat = scipy.io.loadmat(file)
    sr = mat["fs"][0][0]
    vowels = mat["v"][0]

    a = vowels[0].squeeze()
    e = vowels[1].squeeze()
    i = vowels[2].squeeze()

    transform_vowel(a, i, sr)


if __name__ == "__main__":
    main()
