import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def H_s():
    T = 1
    Omega = 0.65 / T
    return [Omega], [1, Omega]


def H_z():
    return [0.245, 0.245], [1, -0.51]


def main():
    b, a = H_z()
    w, H = sig.freqz(b, a)

    ax = plt.subplot()
    ax.plot(w / (2 * np.pi), np.abs(H))
    ax.set_title("Discrete filter")
    ax.set_xlabel("f")
    ax.set_ylabel("|H(f)|")
    plt.show()

    b, a = H_s()
    w, H = sig.freqs(b, a)

    ax = plt.subplot()
    ax.plot(w / (2 * np.pi), np.abs(H))
    ax.set_title("Analog filter")
    ax.set_xlabel("f")
    ax.set_ylabel("|H(f)|")
    plt.show()


if __name__ == "__main__":
    main()
