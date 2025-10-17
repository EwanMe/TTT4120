import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import scipy.signal

N_x = 28


def X_f():
    N_x = 1000
    f = np.arange(0, N_x, 0.001)
    return (1 - (0.9 * np.e) ** (-2j * np.pi * f)) ** N_x / (
        1 - 0.9 * np.e ** (2j * np.pi * f)
    )


def plot_signal_freq_rsp(X, title: str, yax: str, xax: str):
    plt.subplot()
    plt.stem(X)
    plt.title(title)
    plt.ylabel(yax)
    plt.xlabel(xax)
    plt.show()


def x():
    n = np.arange(0, N_x - 1)
    return 0.9**n


def subtask_b():
    plt.subplot()
    plt.plot(np.abs(X_f()), scalex=0.001)
    plt.title("title")
    plt.ylabel("|X(f)|")
    plt.xlabel("f")
    plt.show()

    for div in [4, 2, 1, 1 / 2]:
        fft_len = int(N_x * (1 / div))
        X_k = scipy.fft.fft(x(), n=fft_len)
        plot_signal_freq_rsp(
            X_k, f"FFT length: N_x/{div}={fft_len}", "X(k)", "k"
        )


def main():
    subtask_b()


if __name__ == "__main__":
    main()
