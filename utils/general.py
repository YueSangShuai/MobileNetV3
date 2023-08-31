import math


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


if __name__ == "__main__":
    temp = one_cycle()
    print(one_cycle())
