import math
import sounddevice as sd


def generate_sequence(duration: int, f_1: float, F_s: int) -> list[float]:
    amplitude = 1
    return [
        amplitude * math.cos(2 * math.pi * f_1 * n)
        for n in range(0, duration * F_s)
    ]


def subtask_c() -> None:
    duration = 4
    f_1 = 0.3
    sample_rates = [1000, 3000, 6000]

    for sr in sample_rates:
        seq = generate_sequence(duration, f_1, sr)
        sd.play(seq, sr)
        sd.wait()


def subtask_d() -> None:
    duration = 4
    F_s = 8000
    freqs = [1000, 3000, 6000]

    for F_1 in freqs:
        f_1 = F_1 / F_s
        seq = generate_sequence(duration, f_1, F_s)
        print(f"f_1: {f_1}")
        sd.play(seq, F_s)
        sd.wait()


def main() -> None:
    subtask_c()
    subtask_d()


if __name__ == "__main__":
    main()
