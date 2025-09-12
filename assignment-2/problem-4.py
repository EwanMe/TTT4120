import sounddevice as sd
import numpy as np


def x(n, fs):
    return np.cos((2000 / fs) * np.pi * n)


def subtask_b():
    fs = 4000
    duration = 1
    sd.play([x(n, fs) for n in range(0, duration * fs)], fs)
    sd.wait()

    fs = 1500
    sd.play([x(n, fs) for n in range(0, duration * fs)], fs)
    sd.wait()


def main() -> None:
    subtask_b()


if __name__ == "__main__":
    main()
