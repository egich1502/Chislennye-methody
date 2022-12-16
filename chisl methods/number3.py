import copy
import numpy as np
from collections.abc import Sequence, MutableSequence


def gauss(matrix):
    n = len(matrix)
    matrix_clone = copy.deepcopy(matrix)

    for k in range(n):
        for i in range(n + 1):
            matrix_clone[k][i] = matrix_clone[k][i] / matrix[k][k]
        for i in range(k + 1, n):
            K = matrix_clone[i][k] / matrix_clone[k][k]
            for j in range(n + 1):
                matrix_clone[i][j] = matrix_clone[i][j] - matrix_clone[k][j] * K
        for i in range(n):
            for j in range(n + 1):
                matrix[i][j] = matrix_clone[i][j]

    for k in range(n - 1, -1, -1):
        for i in range(n, -1, -1):
            matrix_clone[k][i] = matrix_clone[k][i] / matrix[k][k]
        for i in range(k - 1, -1, -1):
            K = matrix_clone[i][k] / matrix_clone[k][k]
            for j in range(n, -1, -1):
                matrix_clone[i][j] = matrix_clone[i][j] - matrix_clone[k][j] * K

    answer = []
    for i in range(n):
        answer.append(matrix_clone[i][n])
    return answer


def bubble_max_row(m, col):
    """Replace m[col] row with the one of the underlying rows with the modulo greatest first element.
    :param m: matrix (list of lists)
    :param col: index of the column/row from which underlying search will be launched
    :return: None. Function changes the matrix structure.
    """
    max_element = m[col][col]
    max_row = col
    for i in range(col + 1, len(m)):
        if abs(m[i][col]) > abs(max_element):
            max_element = m[i][col]
            max_row = i
    if max_row != col:
        m[col], m[max_row] = m[max_row], m[col]


def gauss_mod(m):
    """Solve linear equations system with gaussian method.
    :param m: matrix (list of lists)
    :return: None
    """
    n = len(m)
    # forward trace
    for k in range(n - 1):
        bubble_max_row(m, k)
        for i in range(k + 1, n):
            div = m[i][k] / m[k][k]
            m[i][-1] -= div * m[k][-1]
            for j in range(k, n):
                m[i][j] -= div * m[k][j]

    # check modified system for nonsingularity
    if is_singular(m):
        print('The system has infinite number of answers...')
        return

    # backward trace
    x = [0 for i in range(n)]
    for k in range(n - 1, -1, -1):
        x[k] = (m[k][-1] - sum([m[k][j] * x[j] for j in range(k + 1, n)])) / m[k][k]

    # Display results
    return x


def is_singular(m):
    """Check matrix for nonsingularity.
    :param m: matrix (list of lists)
    :return: True if system is nonsingular
    """
    for i in range(len(m)):
        if not m[i][i]:
            return True
    return False


def jacobi(
        A: Sequence[Sequence[float]],
        b: Sequence[float],
        eps: float = 0.001,
        x_init: MutableSequence[float] | None = None):
    """
    метод Якоби для A*x = b (*)

    :param A: Матрица (*) слева

    :param b: известный вектор (*) справа

    :param eps: точность

    :param x_init: начальное приближение

    :return: приблизительное значения х (*)
    """

    def summa(a: Sequence[float], x: Sequence[float], j: int) -> float:
        S: float = 0
        for i, (m, y) in enumerate(zip(a, x)):
            if i != j:
                S += m * y
        return S

    def norm(x: Sequence[float], y: Sequence[float]) -> float:
        return max((abs(x0 - y0) for x0, y0 in zip(x, y)))

    if x_init is None:
        y = [a / A[i][i] for i, a in enumerate(b)]
    else:
        y = x.copy()

    x: list[float] = [-(summa(a, y, i) - b[i]) / A[i][i]
                      for i, a in enumerate(A)]
    k = 0

    while norm(y, x) > eps:
        k += 1
        for i, elem in enumerate(x):
            y[i] = elem
        for i, a in enumerate(A):
            x[i] = -(summa(a, y, i) - b[i]) / A[i][i]
    return x, k


def seidel(A, b, eps):
    n = len(A)
    x = np.zeros(n)  # zero vector
    i = 0

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = np.linalg.norm(x_new - x) <= eps
        x = x_new
        i += 1

    return x, i


def b_jacoby(matrix):
    n = len(matrix)
    d = np.zeros((n, n))
    r_l = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                d[i][j] = matrix[i][j]
            else:
                r_l[i][j] = matrix[i][j]

    d = np.linalg.inv(d)

    return np.matmul(d, r_l)


def b_seidel(matrix):
    n = len(matrix)
    d_l = np.zeros((n, n))
    r = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j <= i:
                d_l[i][j] = matrix[i][j]
            else:
                r[i][j] = matrix[i][j]
    d_l = np.linalg.inv(d_l) * -1

    return np.matmul(d_l, r)


matrix_a = [
    [-0.15, 1.65, 1.15, 4.8],
    [0.95, -0.5, 0.1, 1.7],
    [-0.35, -0.25, -0.7, -3.05],
]

matrix_a_iter = [
                [0.95, -0.5, 0.1],
                [-0.15, 1.65, 1.15],
                [-0.35, -0.25, -0.7]
]
free_b_iter = [
                1.7,
                4.8,
                -3.05
]

epsilon = 0.5e-4

print(gauss(matrix_a), end=' - гаусса\n')
print(gauss_mod(matrix_a), end=' - гаусса с выбором главного элемента\n')
print(jacobi(matrix_a_iter, free_b_iter, epsilon),  end=' - якоби\n')
print(seidel(matrix_a_iter, free_b_iter, epsilon), end=' - гаусса-зейделя\n')
