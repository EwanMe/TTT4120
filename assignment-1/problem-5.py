import numpy as np
import matplotlib.pyplot as plt


def x():
    n_1 = np.arange(0, 3)
    return n_1 + 1


def h_1():
    return [1, 1, 1]


def h_2():
    n_2 = np.arange(0, 11)
    return 0.9**n_2


def exercise_b():
    y_1 = np.convolve(x(), h_1())
    plt.stem(y_1)
    plt.show()

    y_2 = np.convolve(y_1, h_2())
    plt.stem(y_2)
    plt.show()


def exercise_d():
    y_1 = np.convolve(x(), h_2())
    plt.stem(y_1)
    plt.show()

    y_2 = np.convolve(y_1, h_1())
    plt.stem(y_2)
    plt.show()


def main() -> None:
    exercise_b()
    exercise_d()


if __name__ == "__main__":
    main()
