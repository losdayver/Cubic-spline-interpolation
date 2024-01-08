import numpy as np
import matplotlib.pyplot as plt
import math


def generateSplineMatrix(points: list) -> np.array:
    result_array = []

    num_sectors = len(points) - 1

    # Заполнение строк, отвечающих за равенство точек сплайна точкам табличной функции
    for i in range(num_sectors):
        row = [0 for j in range(num_sectors * 4)]

        row[i * 4] = pow(points[i][0], 3)
        row[i * 4 + 1] = pow(points[i][0], 2)
        row[i * 4 + 2] = points[i][0]
        row[i * 4 + 3] = 1

        result_array.append(row)

        row = [0 for j in range(num_sectors * 4)]

        row[i * 4] = pow(points[i + 1][0], 3)
        row[i * 4 + 1] = pow(points[i + 1][0], 2)
        row[i * 4 + 2] = points[i + 1][0]
        row[i * 4 + 3] = 1

        result_array.append(row)

    # Заполнение строк, отвечающих за равенство первых производных
    # в точках табличной функции между смежными участками
    for i in range(1, len(points) - 1):
        row = [0 for j in range(num_sectors * 4)]

        row[(i - 1) * 4] = 3 * pow(points[i][0], 2)
        row[(i - 1) * 4 + 1] = 2 * points[i][0]
        row[(i - 1) * 4 + 2] = 1

        row[(i - 1) * 4 + 4] = -3 * pow(points[i][0], 2)
        row[(i - 1) * 4 + 5] = -2 * points[i][0]
        row[(i - 1) * 4 + 6] = -1

        result_array.append(row)

    # Заполнение строк, отвечающих за равенство вторых производных
    # в точках табличной функции между смежными участками
    for i in range(1, len(points) - 1):
        row = [0 for j in range(num_sectors * 4)]

        row[(i - 1) * 4] = 6 * points[i][0]
        row[(i - 1) * 4 + 1] = 2

        row[(i - 1) * 4 + 4] = -6 * points[i][0]
        row[(i - 1) * 4 + 5] = -2

        result_array.append(row)

    # Заполнение строк, отвечающих за равенство вторых производных нулю в крайних точках отрезка
    row = [0 for j in range(num_sectors * 4)]
    row[0] = 6 * points[0][0]
    row[1] = 2
    result_array.append(row)

    row = [0 for j in range(num_sectors * 4)]
    row[num_sectors * 4 - 3] = 6 * points[-1][0]
    row[num_sectors * 4 - 2] = 2
    result_array.append(row)

    return np.array(result_array)


def generateAnswersVector(points: list) -> np.array:
    B = []

    for i in range(1, len(points)):
        B.append(points[i - 1][1])
        B.append(points[i][1])

    return np.array(B + [0] * len(B))


def plotSpline(points: list, X: np.array):

    for i in range(1, len(points)):
        x = np.linspace(points[i - 1][0], points[i][0])

        start_index = (i - 1) * 4

        y = X[start_index]*(x**3)+X[start_index + 1]*(x**2) + \
            X[start_index + 2]*x+X[start_index + 3]

        plt.plot(x, y)

        plt.plot(points[i][0], points[i][1], 'go')

    plt.plot(points[0][0], points[0][1], 'go')


def getSplineValueAt(x: float, points: list, X: np.array) -> list:
    start_index = 1

    for i, point in enumerate(points[1:]):
        if point[0] >= x:
            start_index = min(i * 4, X.shape[0] - 4)
            break

    return X[start_index]*(x**3)+X[start_index + 1]*(x**2) + \
        X[start_index + 2]*x+X[start_index + 3]


def plotFunction(start_x: float, end_x: float, f, label='original function'):
    x = np.linspace(start_x, end_x)

    y = f(x)

    plt.plot(x, y, label=label)


def generatePoints(start_x: float, end_x: float, num_points: int, f) -> list:
    h = (end_x - start_x) / (num_points - 1)

    x = start_x
    points = []

    for i in range(num_points):
        points.append([x, f(x)])
        x += h

    return points


def calculateError(start_x: float, end_x: float, h: float, f1, f2) -> tuple:
    x = start_x

    max_error = 0
    x_max = 0

    while x <= end_x:
        temp = abs(f1(x) - f2(x))

        if temp > max_error:
            max_error = temp
            x_max = x

        x += h

    return (x_max, max_error)


def runTest(a: float, b: float, min_num_points: int, max_num_points: int, f, f_str: str):
    print(f_str)
    print('num_points\terror')

    for num_points in range(min_num_points, max_num_points):
        points = generatePoints(a, b, num_points, f)

        A = generateSplineMatrix(points)

        B = generateAnswersVector(points)

        X = np.dot(np.linalg.inv(A), B)

        def decoratedGetSplineValueAt(x):
            return getSplineValueAt(x, points, X)

        x_error, error = calculateError(
            a, b, 0.00001, f, decoratedGetSplineValueAt)

        print(f'{num_points}\t{error}')


# runTest(-2, 2, 10, 30, lambda x: np.cos(x), 'cos(x)')

a = 1
b = 10
num_points = 5
def func(x): return 1/x


points = generatePoints(a, b, num_points, func)

A = generateSplineMatrix(points)

B = generateAnswersVector(points)

X = np.dot(np.linalg.inv(A), B)

plotFunction(a, b, func)

plotSpline(points, X)


def decoratedGetSplineValueAt(x):
    global points, X
    return getSplineValueAt(x, points, X)


x_error, error = calculateError(
    a, b, 0.00001, func, decoratedGetSplineValueAt)

func_at_error = func(x_error)
spline_at_error = decoratedGetSplineValueAt(x_error)

plt.vlines(x=x_error, ymin=min(func_at_error, spline_at_error),
           ymax=max(func_at_error, spline_at_error), color='b',
           label='error')

plt.plot(x_error, func_at_error, 'ro', label='error point')
plt.plot(x_error, spline_at_error, 'ro')

plt.text(0.01, 0.01, f'max error = {error} at {x_error}',
         transform=plt.gca().transAxes)

plt.legend(loc='best')
plt.grid(True, linestyle='dashed')
plt.show()
