import matplotlib.pyplot as plt
import numpy as np


def Y(omega, M):
    return np.sin(omega * (M + 1 / 2)) / np.sin(omega / 2)


def subtask_b() -> None:
    M = 10
    plt.plot([Y(x, M) for x in np.linspace(-np.pi, np.pi, 1000)])
    plt.show()


def x(n):
    if n == 0:
        return 2
    if abs(n) == 1:
        return 1
    return 0


def z(n):
    N = 4

    r = range(-1000, 1000)
    return sum([x(n - l * N) for l in r])


def main() -> None:
    subtask_b()


if __name__ == "__main__":
    main()
