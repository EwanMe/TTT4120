import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.io
import scipy.signal


def plot(f, title, y_label, x_label, x_vals=None):
    if x_vals is None:
        x_vals = np.arange(len(f))

    plt.title(title)
    ax1 = plt.subplot()
    ax1.figure.set_size_inches(15, 5)
    ax1.stem(x_vals, f)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax1.legend()
    ax1.grid(True)
    plt.show()


def subtask_a(x, y):
    plot(x, "Problem 2 (a): x[n]", "x[n]", "n")
    plot(y, "Problem 2 (a): y[n]", "y[n]", "n")


def subtask_b(x, y):
    r_yx = scipy.signal.correlate(y, x)
    plot(
        r_yx,
        "Problem 2 (b): r_yx[l]",
        "r_yx[l]",
        "l",
        np.arange(-len(r_yx) / 2, len(r_yx) / 2),
    )


def subtask_c(x, y):
    r_yx = np.convolve(y, np.flip(x))
    plot(
        r_yx,
        "Problem 2 (c): r_yx[l]",
        "r_yx[l]",
        "l",
        np.arange(-len(r_yx) / 2, len(r_yx) / 2),
    )


def main():
    signals = scipy.io.loadmat("signals.mat")
    x = signals["x"][0]
    y = signals["y"][0]
    subtask_a(x, y)
    subtask_b(x, y)
    subtask_c(x, y)


if __name__ == "__main__":
    main()
