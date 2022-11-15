import math

epsilon = 0.5 * 10 ** -5


def f(x):
    return math.log10(x) - 0.18 / x


def f_prime(x):
    return 1 / (math.log(10, math.e) * x) + 9 / (50 * x ** 2)


def fi(x):
    return 10 ** (0.18 / x)


def fi_prime(x):
    return -((9 * math.log(10) * 10 ** (9 / (50 * x) - 1)) / (5 * x * x))


def dichotomy(func):
    a = 1
    b = 2
    i = 0
    while b - a > epsilon:
        c = (a + b) / 2
        if (func(b) * func(c)) < 0:
            a = c
        else:
            b = c
        i += 1
    return (a + b) / 2, i


def newton(func, func_prime, x0, kmax=1e3):
    x, x_prev, i = x0, x0 + 2 * epsilon, 0

    while abs(func(x)) >= epsilon and i < kmax:
        x, x_prev, i = x - func(x) / func_prime(x), x, i + 1

    return x, i


def newton_mod(func, func_prime, x0, kmax=1e3):
    x, x_prev, i = x0, x0 + 2 * epsilon, 0

    while abs(func(x)) >= epsilon and i < kmax:
        x, x_prev, i = x - func(x) / func_prime(x0), x, i + 1

    return x, i


def chord(func, x0, kmax=1e3):
    x, i = x0 + epsilon, 0

    while abs(func(x)) >= epsilon and i < kmax:
        x, i = x - (func(x) * (x - x0)) / (func(x) - func(x0)), i + 1
    return x, i


def moving_chord(func, x0, kmax=1e3):
    x, x_prev, i = x0, x0 + 2 * epsilon, 0

    while abs(func(x)) >= epsilon and i < kmax:
        x, x_prev, i = x - (func(x) * (x - x_prev)) / (func(x) - func(x_prev)), x, i + 1
    return x, i


def simple_iteration(func, iter_func, x0):
    x, i = x0, 0

    while abs(func(x)) >= epsilon and i < 1e3:
        x, i = iter_func(x), i + 1
    return x, i


def main():
    print('дихотомия', dichotomy(f))
    print('ньютон', newton(f, f_prime, 1))
    print('ньютон модифицированный', newton_mod(f, f_prime, 1))
    print('хорд', chord(f, 1))
    print('подвижных хорд', moving_chord(f, 1))
    print('простой итерации', simple_iteration(f, fi, 1))


if __name__ == '__main__':
    main()
