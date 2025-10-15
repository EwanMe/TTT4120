import matplotlib.pyplot as plt
import numpy as np

N_x = 28
NFFT = N_x // 2


def plot_signal_freq_rsp(X, f, title: str):
    mag = np.abs(X) / N_x
    plt.figure(figsize=(12, 5))
    plt.subplot()
    plt.plot(f[:NFFT], mag[:NFFT])
    plt.title(title)
    plt.xlabel("f")
    plt.show()


def x():
    n = np.arange(0, N_x - 1)
    return 0.9**n


def main():

    f = np.arange(NFFT) / NFFT
    X = np.fft.fft(x(), n=NFFT)
    plot_signal_freq_rsp(X, f, "Test")


if __name__ == "__main__":
    main()
