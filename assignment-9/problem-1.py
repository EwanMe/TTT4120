import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal as sig


def amp_to_dB(amp: npt.NDArray) -> npt.NDArray:
    return 20 * np.log10(amp)


def h(n: npt.NDArray, f_c: float, w: npt.NDArray) -> npt.NDArray[np.float64]:
    if len(n) != len(w):
        raise ValueError("n and w must be of equal length")

    M = (len(n) - 1) // 2
    return 2 * f_c * np.sinc(2 * f_c * (n - M)) * w


def subtask_d():
    N = 31
    n = np.arange(N)
    cutoff_freq = 0.2

    for name, window in {
        "Rectangular": np.ones(N),
        "Hamming": np.hamming(N),
    }.items():
        w, H = sig.freqz(h(n, cutoff_freq, window), worN=1024)

        ax = plt.subplot()
        ax.set_title(name)
        ax.plot(w / (2 * np.pi), amp_to_dB(np.abs(H)))
        ax.set_ylabel("dB")
        ax.set_xlabel("f")
        plt.show()


def main():
    subtask_d()


if __name__ == "__main__":
    main()
