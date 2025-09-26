from typing import Any

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

    return d, g

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


def plot_signal(x, title, y_label, x_label):
    plt.title(title)
    ax1 = plt.subplot()
    ax1.figure.set_size_inches(15, 5)
    ax1.plot(x)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax1.grid(True)
    plt.show()


def plot_poles_and_zeroes(poles: np.ndarray, zeroes: np.ndarray, title: str):
    fig, ax1 = plt.subplots()

    # plot circle
    theta = np.linspace(-np.pi, np.pi, 1000)
    plt.title(title)
    ax1.plot(np.sin(theta), np.cos(theta), "--k")
    ax1.set_aspect(1)
    # plot poles and zeros
    ax1.plot(np.real(poles), np.imag(poles), "Xb", label="Poles")
    ax1.plot(np.real(zeroes), np.imag(zeroes), "or", label="Zeros")
    ax1.set_xlabel("Real part")
    ax1.set_ylabel("Imaginary part")
    ax1.grid(True)
    plt.show()


def plot_frequency_response(w, h, title):
    plt.title(title)
    ax2 = plt.subplot()
    ax2.plot(w, h)
    ax2.set_ylabel("Magnitude")
    ax2.set_xlabel("ω")
    ax2.legend()
    ax2.grid(True)
    # plt.xticks(np.arange(0, 0.51, 0.02))
    # plt.xticks(
    #     np.arange(0, np.pi + np.pi / 2, step=(np.pi / 2)),
    #     ["0", "π/2", "π"],
    # )
    plt.show()


def subtask_b_c_d(
    d: np.ndarray[tuple[Any, ...], np.dtype[np.float64]],
    g: np.ndarray[tuple[Any, ...], np.dtype[np.float64]],
):
    # subtask b
    b = np.poly([-1, 1])
    r = 0.99
    w1 = 2 * np.pi * 0.04j
    w2 = 2 * np.pi * 0.1j
    ax = np.poly([r * np.e ** (-w1), r * np.e**w1])
    ay = np.poly([r * np.e ** (-w2), r * np.e**w2])

    zeroes = np.roots(b)
    x_poles = np.roots(ax)

    plot_poles_and_zeroes(x_poles, zeroes, "h_x poles and zeroes")

    wx, hx = scipy.signal.freqz(b, ax)
    plot_frequency_response(wx, hx, "|H_x(f)|")

    y_poles = np.roots(ay)
    plot_poles_and_zeroes(y_poles, zeroes, "h_y poles and zeroes")
    wy, hy = scipy.signal.freqz(b, ay)
    plot_frequency_response(wy, hy, "|H_y(f)|")

    # subtask c
    qx = scipy.signal.lfilter(b, ax, g)
    qy = scipy.signal.lfilter(b, ay, g)
    plot_signal(qx, "q_x[n]", "q_x[n]", "n")
    plot_signal(qy, "q_y[n]", "q_y[n]", "n")

    wqx, hqx = scipy.signal.freqz(b, ax)
    plot_frequency_response(wqx, hqx, "|Q_x(f)|")
    wqy, hqy = scipy.signal.freqz(b, ay)
    plot_frequency_response(wqy, hqy, "|Q_y(f)|")

    # subtask d


def main():
    d, g = subtask_a()
    subtask_b_c_d(d, g)


if __name__ == "__main__":
    main()
