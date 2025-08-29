import math
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def generate_sequence(duration: int, F_s: int) -> list[float]:
    A = 1
    f_1 = 440
    return [
        A * math.cos(2 * math.pi * f_1 * n) for n in range(0, duration * F_s)
    ]


def draw(sequence: list[float]) -> None:
    x = np.linspace(0, 2 * np.pi, 100)
    plt.plot([n for n in range(0, 100)], sequence[0:100])
    plt.grid(True)
    plt.show()


def main() -> None:
    duration = 4
    sample_rate = 6000
    seq = generate_sequence(duration, sample_rate)

    duration = 4.0
    frequency = 30
    sample_rate = 1000

    # Generate a sine wave (you could use cosine too)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    print(wave)
    # Play the sound
    sd.play(wave, sample_rate)
    sd.wait()  # Wait until sound is done playing


if __name__ == "__main__":
    main()
