import numpy as np
import numpy.typing as npt
import scipy.signal as sig
import scipy.io.wavfile as wav
import sounddevice as sd
import time


def normalize(x) -> npt.NDArray[np.float64]:
    return x / np.max(x)


def resample(x: npt.NDArray[np.float64], D: int) -> npt.NDArray[np.float64]:
    """
    scipy.signal.resample doesn't allow for doing a naive downsampling
    without filtering, so here we just pick every D-th sample directly.

    """
    return x[::D]


def test_decimation(
    x: npt.NDArray[np.float64], decimation_factor: int, sample_rate: int
) -> None:
    print("Original....")
    sd.play(normalize(x), sample_rate)
    sd.wait()

    time.sleep(1)
    print("Decimated...")
    xr = sig.decimate(x, decimation_factor)
    sd.play(normalize(xr), sample_rate / decimation_factor)
    sd.wait()

    time.sleep(1)
    print("Downsampled without filter...")
    xr = resample(x, decimation_factor)
    sd.play(normalize(xr), sample_rate / decimation_factor)
    sd.wait()


def main():
    Fs = 6000
    f1 = 900 / Fs
    f2 = 2000 / Fs
    D = 2

    n = np.arange(Fs)  # One second
    x1 = np.cos(2 * np.pi * f1 * n)
    x2 = np.cos(2 * np.pi * f2 * n)
    x = x1 + x2

    test_decimation(x, D, Fs)

    fs, x = wav.read("Dolly.wav")

    # Merge into mono
    if x.ndim > 1:
        x = x.mean(axis=1)
    test_decimation(x, D, fs)


if __name__ == "__main__":
    main()
