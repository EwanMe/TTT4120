import matplotlib.pyplot as plt
import numpy as np


N_h = 9


def x():
    n = np.arange(0, N_h - 1)
    return 0.9**n


def h():
    return np.ones(N_h - 1)


def stem(title, y_label, x_label, *functions):
    plt.subplot()
    plt.title(title)

    i = 0
    for f in functions:
        print(f)
        if len(functions) == 1:
            plt.stem(f)
        else:
            plt.stem(f, label="name", linefmt=f"C{i}-")
            i += 1
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def y():
    return np.convolve(x(), h())


def subtask_a():
    plt.subplot()
    plt.title("Problem 2a)")
    plt.stem(y())
    plt.ylabel("y[n]")
    plt.xlabel("n")
    plt.show()


def subtask_b():
    N_y = 16
    for nfft in [N_y // 4, N_y // 2, N_y, 2 * N_y]:
        X = np.fft.fft(x(), nfft)
        H = np.fft.fft(h(), nfft)

        y_fft = np.fft.ifft(X * H, nfft)

        plt.subplot()
        plt.title(f"Problem 2b): FFT length={nfft}")
        plt.stem(y(), label="y[n]", linefmt="C0-")
        plt.stem(y_fft, label="y_ifft[n]", linefmt="C1-")
        plt.ylabel("y[n]")
        plt.xlabel("n")
        plt.show()


def main():
    subtask_a()
    subtask_b()


if __name__ == "__main__":
    main()
