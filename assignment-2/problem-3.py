import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def plot(w, func, title, ylabel, xlabel):
    plt.title(title)
    mag = plt.subplot()
    mag.plot(w, func)
    mag.set_ylabel(ylabel)
    mag.set_xlabel(xlabel)
    mag.grid(True)
    plt.xticks(
        np.arange(0, np.pi + np.pi / 2, step=(np.pi / 2)),
        ["0", "π/2", "π"],
    )
    plt.show()


def freq_resp(b, a, title):
    w, h = scipy.signal.freqz(b, a)
    plot(w, np.abs(h), title, "|H(ω)|", "ω")
    plot(w, np.unwrap(np.angle(h)), title, "Θ(ω)", "ω")


def subtask_c():
    b = [1, 2, 1]
    a = [1]
    freq_resp(b, a, "y[n]=x[n]+2x[n-1]+x[n-2]")

    b = [1]
    a = [1, 0.9]
    freq_resp(b, a, "y[n]=-0.9y[n-1]+x[n]")


def main():
    subtask_c()


if __name__ == "__main__":
    main()
