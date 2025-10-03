import matplotlib.pyplot as plt
import numpy as np


def x(n, a):
    return a**n


def s_xx(f, a):
    return 1 / (1 - 2 * a * np.cos(2 * np.pi * f) + a**2)


def r_xx(l, a):
    return (a ** np.abs(l)) / (1 - a**2)


def plot(functions: list, names: list, y_label, x_label):
    ax1 = plt.subplot()
    ax1.figure.set_size_inches(15, 5)

    for i in range(len(functions)):
        ax1.stem(functions[i], label=names[i], linefmt=f"C{i}-")

    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax1.legend()
    ax1.grid(True)
    plt.show()


def main():
    n = np.arange(50)
    l = np.arange(-50, 50)
    f = np.arange(-0.5, 0.5, step=0.01)

    a_n = [0.5, 0.9, -0.9]
    names = ["a=0.5", "a=0.9", "a=-0.9"]

    plot([x(n, a) for a in a_n], names, "x[n]", "n")
    plot([s_xx(f, a) for a in a_n], names, "s_xx[f]", "f")
    plot([r_xx(l, a) for a in a_n], names, "r_xx[l]", "l")


if __name__ == "__main__":
    main()
