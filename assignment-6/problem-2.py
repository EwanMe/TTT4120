import matplotlib.pyplot as plt
import numpy as np


N_h = 9


def x():
    n = np.arange(0, N_h - 1)
    return 0.9**n


def h():
    return np.ones(N_h - 1)


def subtask_a():
    plt.subplot()
    plt.title("Problem 2a)")
    plt.stem(np.convolve(x(), h()))
    plt.ylabel("y[n]")
    plt.xlabel("n")
    plt.show()


def main():
    subtask_a()


if __name__ == "__main__":
    main()
