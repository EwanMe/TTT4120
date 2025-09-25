import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def subtask_a():
    L = 500
    A_x = A_y = 0.25
    f_x = 0.04
    f_y = 0.10
    NFFT = 2048
    n = np.arange(L)
    d = A_x * np.cos(2 * np.pi * f_x * n) + A_y * np.cos(2 * np.pi * f_y * n)
    e = np.random.normal(size=L)
    g = d + e

    plt.title("d[n]")
    ax1 = plt.subplot()
    ax1.figure.set_size_inches(15, 5)
    ax1.plot(d)
    ax1.set_ylabel("d[n]")
    ax1.set_xlabel("n")
    ax1.grid(True)
    plt.show()

    plt.title("g[n]")
    ax2 = plt.subplot()
    ax2.figure.set_size_inches(15, 5)
    ax2.plot(g)
    ax2.set_ylabel("g[n]")
    ax2.set_xlabel("n")
    ax2.grid(True)
    plt.show()

    D = np.fft.fft(d, n=NFFT)
    f = np.arange(NFFT) / NFFT

    magD = np.abs(D) / L

    plt.figure(figsize=(12, 5))
    plt.subplot()
    plt.plot(f[: NFFT // 2], magD[: NFFT // 2])
    plt.title("Magnitude |D(f)|")
    plt.xlabel("f (cycles/sample)")
    plt.xticks(np.arange(0, 0.51, 0.02))
    plt.show()

    G = np.fft.fft(g, n=NFFT)
    magG = np.abs(G) / L
    plt.figure(figsize=(12, 5))
    plt.subplot()
    plt.plot(f[: NFFT // 2], magG[: NFFT // 2])
    plt.title("Magnitude |G(f)|")
    plt.xlabel("f (cycles/sample)")
    plt.show()


def subtask_b():
    pass


def main():
    subtask_a()
    subtask_b()


if __name__ == "__main__":
    main()
