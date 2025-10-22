import matplotlib.pyplot as plt
import numpy as np
import scipy.fft


def X_f(N):
    f = np.linspace(0, 1, 10000)
    return f, (1 - (0.9 * np.exp(-2j * np.pi * f)) ** N) / (
        1 - 0.9 * np.exp(-2j * np.pi * f)
    )


def plot_dft(X, title: str, yax: str, xax: str):
    plt.subplot()
    plt.stem(X)
    plt.title(title)
    plt.ylabel(yax)
    plt.xlabel(xax)
    plt.show()


def plot_dtft(N):
    f, X = X_f(N)
    plt.plot(f, np.abs(X))
    plt.title("DTFT")
    plt.ylabel("|X(f)|")
    plt.xlabel("f")
    plt.show()


def x(N_x):
    n = np.arange(0, N_x - 1)
    return 0.9**n


def subtask_b():
    N_x = 28
    plot_dtft(N_x)
    for div in [4, 2, 1, 1 / 2]:
        fft_len = int(N_x * (1 / div))
        X_k = scipy.fft.fft(x(N_x), n=fft_len)
        plot_dft(X_k, f"DFT: FFT length: N_x/{div}={fft_len}", "X(k)", "k")


def main():
    subtask_b()


if __name__ == "__main__":
    main()
