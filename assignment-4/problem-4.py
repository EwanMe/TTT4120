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

    return g

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
    plt.xlabel("f")
    plt.xticks(np.arange(0, 0.51, 0.02))
    plt.show()

    G = np.fft.fft(g, n=NFFT)
    magG = np.abs(G) / L
    plt.figure(figsize=(12, 5))
    plt.subplot()
    plt.plot(f[: NFFT // 2], magG[: NFFT // 2])
    plt.title("Magnitude |G(f)|")
    plt.xlabel("f")
    plt.show()

    return g


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
    ax2.plot(w / (2 * np.pi), h)
    ax2.set_ylabel("Magnitude")
    ax2.set_xlabel("f")
    ax2.legend()
    ax2.grid(True)
    plt.show()


def plot_signal_freq_rsp(x, title: str):
    L = 500
    NFFT = 2048

    f = np.arange(NFFT) / NFFT
    X = np.fft.fft(x, n=NFFT)
    mag = np.abs(X) / L
    plt.figure(figsize=(12, 5))
    plt.subplot()
    plt.plot(f[: NFFT // 2], mag[: NFFT // 2])
    plt.title(title)
    plt.xlabel("f")
    plt.show()

    return f, mag


def subtask_b_c_d(g: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]):
    # subtask b
    b = np.poly([-1, 1])
    r = 0.99
    w1 = 2 * np.pi * 0.04j
    w2 = 2 * np.pi * 0.1j
    ax = np.poly([r * np.e ** (-w1), r * np.e**w1])
    ay = np.poly([r * np.e ** (-w2), r * np.e**w2])

    zeroes = np.roots(b)
    y_poles = np.roots(ay)
    x_poles = np.roots(ax)

    # plot_poles_and_zeroes(x_poles, zeroes, "h_x poles and zeroes")

    wx, hx = scipy.signal.freqz(b, ax)
    # plot_frequency_response(wx, hx, "|H_x(f)|")

    # plot_poles_and_zeroes(y_poles, zeroes, "h_y poles and zeroes")
    wy, hy = scipy.signal.freqz(b, ay)
    # plot_frequency_response(wy, hy, "|H_y(f)|")

    # subtask c
    qx = scipy.signal.lfilter(b, ax, g)
    qy = scipy.signal.lfilter(b, ay, g)
    # plot_signal(qx, "q_x[n]", "q_x[n]", "n")
    # plot_signal(qy, "q_y[n]", "q_y[n]", "n")

    wqx, hqx = scipy.signal.freqz(b, ax)
    # plot_signal_freq_rsp(qx)
    # plot_frequency_response(wqx, hqx, "|Q_x(f)|")
    wqy, hqy = scipy.signal.freqz(b, ay)
    # plot_signal_freq_rsp(qy)
    # plot_frequency_response(wqy, hqy, "|Q_y(f)|")

    # subtask d
    x_poles_plus_y_poles = np.roots(np.poly(x_poles) + np.poly(y_poles))
    zeroes = [*zeroes, *x_poles_plus_y_poles]
    poles = [*x_poles, *y_poles]
    wq, hq = scipy.signal.freqz(np.poly(zeroes), np.poly(poles))
    plot_frequency_response(wq, hq, "|Q(f)|")

    plot_poles_and_zeroes(poles, zeroes, "q poles and zeroes")

    q = scipy.signal.lfilter(np.poly(zeroes), np.poly(poles), g)
    plot_signal(q, "q[n]", "q[n]", "n")
    plot_signal_freq_rsp(q, "Magnitude of q[n]")


def main():
    g = subtask_a()
    subtask_b_c_d(g)


if __name__ == "__main__":
    main()
