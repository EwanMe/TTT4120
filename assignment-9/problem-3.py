import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def plot_filter(x, f, *, title: str, x_label: str, y_label: str):
    ax = plt.subplot()
    ax.plot(x, f)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


def subtask_b():
    w, H = sig.freqs([1], [1, np.sqrt(2), 1])
    plot_filter(
        w / (2 * np.pi),
        np.abs(H),
        title="Butterworth filter",
        x_label="f",
        y_label="|H(f)|",
    )


def subtask_e():
    b_1 = [0.208]
    a_1 = [1, -1.65, 0.702]
    w, H = sig.freqz(b_1, a_1)
    plot_filter(
        w / (2 * np.pi),
        np.abs(H),
        title="ω = 0.25",
        x_label="f",
        y_label="|H(f)|",
    )

    b_2 = [0.454]
    a_2 = 1, -0.566, 0.183
    w, H = sig.freqz(b_2, a_2)
    plot_filter(
        w / (2 * np.pi),
        np.abs(H),
        title="ω = 1.2",
        x_label="f",
        y_label="|H(f)|",
    )


def main():
    subtask_b()
    subtask_e()


if __name__ == "__main__":
    main()
