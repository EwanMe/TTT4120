import matplotlib.pyplot as plt
import numpy as np
import scipy.fft

N_x = 28


def plot_signal_freq_rsp(X, title: str, yax: str, xax: str):
    plt.subplot()
    plt.plot(X)
    plt.title(title)
    plt.ylabel(yax)
    plt.xlabel(xax)
    plt.show()


def x():
    n = np.arange(0, N_x - 1)
    return 0.9**n


def subtask_b():
    for div in [4, 2, 1, 1 / 2]:
        fft_len = int(N_x * (1 / div))
        X = scipy.fft.fft(x(), n=fft_len)
        plot_signal_freq_rsp(X, f"FFT length: N_x/{div}={fft_len}", "X(f)", "f")


def main():
    subtask_b()


if __name__ == "__main__":
    main()
