import numpy as np
import statistics
import matplotlib.pyplot as plt
import scipy as sp


def Uniform_fun():
    n = 250

    U = np.random.uniform(size=(n, 100000))
    Mean = np.mean(U, axis=0)
    Median = np.median(U, axis=0)
    Mid_range = (np.max(U, axis=0) + np.min(U, axis=0)) / 2
    R = (Mid_range - 1 / 2) * n
    Min_n = n * np.min(U, axis=0)

    x_exp = np.linspace(0, 10, 100000)
    x_lap = np.linspace(-10, 10, 100000)

    fig, axs = plt.subplots(5)
    fig.suptitle('Vertically stacked subplots')
    axs[0].hist(Mean, bins=100, density=True)
    axs[1].hist(Median, bins=100, density=True)
    axs[2].hist(Mid_range, bins=100, density=True)
    axs[3].hist(R, bins=100, density=True)  # rozklad laplace'a -> symetryczny wykladniczy
    axs[3].plot(x_lap, np.exp(-2 * abs(x_lap)))
    axs[4].hist(Min_n, bins=100, density=True)
    axs[4].plot(x_exp, np.exp(-x_exp))

    plt.show()


def Laplace_dist():
    # p(x) = l/2 * exp(-l*abs(x))
    Exp = np.random.exponential(size=100000)
    U = np.random.uniform(size=100000)

    Coin_flip = [1 if u > 0.5 else -1 for u in U]

    plt.hist(Exp * Coin_flip, bins=100, density=True)
    plt.show()


def Normal_dist():
    b = np.sqrt(2 / np.exp(1))
    U = np.random.uniform(size=100000)
    V = np.random.uniform(low=-b, high=b, size=100000)
    X = []
    U_good = []
    V_good = []
    j = 0
    for i in range(100000):
        if (V[i] / U[i]) ** 2 < -4 * np.log(U[i]):
            X.append(V[i] / U[i])
            U_good.append(U[i])
            V_good.append(V[i])
            j += 1

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].hist(X, bins=100, density=True)
    axs[1].scatter(U_good, V_good)

    plt.show()
    print(j / 100000)


def Normal_dist_two_dim():
    V = np.array([[2, 1], [1, 1]])
    A = np.linalg.cholesky(V)
    print(np.dot(A, A.T.conj()))
    N = np.random.normal(size=(2, 100000))

    X = np.dot(A, N)
    plt.scatter(X[0], X[1], marker='.', alpha=0.05)
    plt.show()


# zadanie 1.6 ze skryptu
# dystrybunta y = 1 - y^(-n-1)
def zad_1_6(n=10):
    N = 1000000
    U = np.random.uniform(size=N)
    Y = U ** (-1 / (n + 1))
    X = np.random.exponential(scale=Y, size=N)
    plt.hist(X, bins=100, density=True)
    plt.show()


# dwuwymiarowy rozklad cauchyiego
# Y1 i Y2 sa losowane z rozkladu t-studenta z 1 stopniem swobody
def zad_1_13():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    xx, yy = np.meshgrid(x, y)
    zz1 = 0.5 / np.pi / (xx ** 2 + yy ** 2 + 1) ** 1.5
    zz2 = 1 / np.pi ** 2 / (1 + xx ** 2) / (1 + yy ** 2)

    plt.contour(xx, yy, zz1, levels=np.linspace(0, 1 / 3, 50))
    plt.contour(xx, yy, zz2, linestyles='dashed', levels=np.linspace(0, 1 / 3, 50))
    plt.show()


def zad_1_13_2():
    k = 1
    X1 = np.random.normal(0, 1, size=100000)
    X2 = np.random.normal(0, 1, size=100000)

    R_1 = np.random.chisquare(k, size=100000)
    R_2 = np.random.chisquare(k, size=100000)

    Y1 = X1 / np.sqrt(R_1)
    Y2 = X2 / np.sqrt(R_2)

    plt.scatter(Y1, Y2)

    Z = np.random.standard_cauchy(size=(10000, 2))
    plt.scatter(Z[:, 0], Z[:, 1], c='red')
    plt.show()


def poisson_process():
    n = 1000000
    X = np.random.exponential(size=n)
    PP = np.cumsum(X)
    Ber = np.random.binomial(p=0.3, n=1, size=n)
    PP_new = []
    for i in range(n):
        if Ber[i]:
            PP_new.append(PP[i])
    PP_new = np.array(PP_new)
    X_new = np.diff(PP_new)
    fig, axs = plt.subplots(2)
    axs[0].hist(X, bins=100, density=True)
    axs[1].hist(X_new, bins=100, density=True)
    plt.show()
    print(np.mean(X))
    print(np.mean(X_new))


def super_position():
    n = 100000
    X = np.random.exponential(4, size=n)
    Y = np.random.exponential(6, size=n)
    PP_X = np.cumsum(X)
    PP_Y = np.cumsum(Y)
    max_time = min(max(PP_X), max(PP_Y))
    PP_SP = np.concatenate([PP_X, PP_Y])
    PP_SP.sort()
    PP_SP = np.array([num for num in PP_SP if num < max_time])
    SP = np.diff(PP_SP)
    fig, axs = plt.subplots(3)
    axs[0].hist(X, bins=200, density=True)
    axs[1].hist(Y, bins=200, density=True)
    axs[2].hist(SP, density=True, bins=200)
    plt.show()


def mc():
    n = 1000000
    A = np.random.exponential(scale=1, size=(1, 2))
    A = [-0.9, 0.9]
    eps = np.random.normal(size=(n, 2))
    X = eps
    for t in range(1, n):
        X[t] += A * X[t - 1]
    """plt.plot(X, linewidth=0.5)
    plt.legend(list(A))"""
    fig, axs = plt.subplots(2)
    axs[0].plot(X[:, 0])
    axs[1].plot(X[:, 1])
    plt.show()


def markov_process():
    a = 2
    b = 3

    X = [2]
    T = [0]
    W = []
    Q = {1: 2, 2: 5}
    for i in range(1, 1000000):
        W.append(np.random.exponential(Q[X[-1]]))
        T.append(T[-1] + W[-1])
        X.append(1 if X[-1] == 2 else 2)

    """plt.step(x=T, y=X)
    plt.show()"""

    sum = 0
    for j in range(1, 1000000):
        if X[j] == 2:
            sum += W[j]
    print(sum / T[-1])


def markov_process_poisson_generation():
    q = 7
    Q = {1: 2 / q, 2: 5 / q}
    W = np.random.exponential(q, size=10000)
    PP = np.cumsum(W)

    X = [1]
    for i in range(1, len(PP)):
        random = np.random.rand()
        if X[-1] == 1:
            X.append(2 if random <= Q[1] else 1)
        elif X[-1] == 2:
            X.append(1 if random <= Q[2] else 2)
    plt.step(PP, X)
    plt.scatter(PP, X)
    plt.show()
    print(X.count(1) / len(X))


def zad_2_13():
    r = 10
    a = 2
    b = 2

    X = [2]
    T = [0]
    W = []
    for i in range(1, 1000000):
        W.append(np.random.exponential(a * X[-1] + b * (r - X[-1])))
        T.append(T[-1] + W[-1])
        random = np.random.rand()
        X.append(X[-1] + 1 if random <= b * (r - X[-1]) / (a * X[-1] + b * (r - X[-1])) else X[-1] - 1)

    """plt.step(T, X)
    plt.show()"""
    perc = {}
    for i in range(0, 11):
        perc[i] = 0
        for j in range(1, 1000000):
            if X[j] == i:
                perc[i] += W[j - 1]
        perc[i] / T[-1]
        print(i, perc[i])

    plt.bar(list(perc.keys()), list(perc.values()))
    plt.show()


def markov_testing():
    a = 2
    size = 200
    X = np.arange(size) + 1
    T = np.zeros(size)

    for i in range(1, T.shape[0]):
        W = np.random.exponential(1 / (a / X[i - 1]))
        T[i] = T[i - 1] + W

    plt.step(T, X)
    plt.show()


def zad_2_11():
    a = 2
    b = 1

    X = [0]
    T = [0]
    W = []
    for i in range(1, 100):
        W.append(np.random.exponential(1 / (a + b * X[-1])))
        T.append(T[-1] + W[-1])
        random = np.random.rand()
        if random <= a / (a + b * X[-1]):
            X.append(X[-1] + 1)
        elif X[-1] == 0:
            X.append(0)
        else:
            X.append(X[-1] - 1)

    plt.step(T, X)
    plt.show()
    perc = {}
    for i in range(0, 11):
        perc[i] = 0
        for j in range(1, 100):
            if X[j] == i:
                perc[i] += W[j - 1]
        perc[i] / T[-1]
        print(i, perc[i])

    """plt.bar(list(perc.keys()), list(perc.values()))
    plt.show()"""


def monte_carlo_zad_3_1_a():
    n = 1000000000
    Z = np.random.normal(size=n)
    Z_greater_than_4 = [int(z > 4) for z in Z]
    theta = sum(Z_greater_than_4) / n
    sigma = np.std(Z_greater_than_4)
    print(f'theta = {theta} +- {sigma * 2 / np.sqrt(n)}')


def monte_carlo_zad_3_1_b():
    n = 1000000
    # w(z) = p(z) / q(z)

    Z = np.random.exponential(size=n) + 4
    W = sp.stats.norm.pdf(Z) / np.exp(-Z + 4)

    print(f'theta = {np.mean(W)} +- {np.std(W) * 2 / np.sqrt(n)}')


def monte_carlo_zad_3_2_a():
    n = 1000
    Z = np.random.standard_cauchy(size=n)
    Z_greater_than_2 = [int(z > 2) for z in Z]
    theta = sum(Z_greater_than_2) / n
    sigma = np.std(Z_greater_than_2)
    print('Metoda podstawowa')
    print(f'theta = {theta} +- {sigma * 2 / np.sqrt(n)}')


def monte_carlo_zad_3_2_b():
    n = 1000
    Z = np.random.standard_cauchy(size=n)
    Z_norm_greater_than_2 = [int(abs(z) > 2) / 2 for z in Z]
    theta = sum(Z_norm_greater_than_2) / n
    sigma = np.std(Z_norm_greater_than_2)
    print('Metoda z symetira')
    print(f'theta = {theta} +- {sigma * 2 / np.sqrt(n)}')


def monte_carlo_zad_3_2_c():
    n = 1000
    U = np.random.uniform(size=n) * 2
    theta = 1 / 2 - sum(2 / (np.pi * (1 + U ** 2))) / n
    sigma = np.std(1 / (np.pi * (1 + U ** 2)))
    print('Metoda z dopelnieniem i rozkladem normalnym')
    print(f'theta = {theta} +- {sigma * 2 / np.sqrt(n)}')


def monte_carlo_zad_3_2_d():
    n = 1000
    U = np.random.uniform(size=n) * 2
    theta = 1 / 2 - sum(1 / (np.pi * (1 + U ** 2))) / n - sum(1 / (np.pi * (1 + (2 - U) ** 2))) / n
    sigma = np.std(1 / (np.pi * (1 + U ** 2)) + 1 / (np.pi * (1 + (2 - U) ** 2)))
    print('Metoda z dopelnieniem i zmienne antytetyczne')
    print(f'theta = {theta} +- {sigma * 2 / np.sqrt(n)}')


def monte_carlo_zad_3_2_e():
    n = 1000
    Z = np.random.standard_cauchy(size=n)
    Z_between_0_and_half = [(abs(z) < 1 / 2) / 2 for z in Z]
    theta = sum(Z_between_0_and_half) / n
    sigma = np.std(Z_between_0_and_half)
    print('Metoda podstawowa + chytry chwyt')
    print(f'theta = {theta} +- {sigma * 2 / np.sqrt(n)}')


def monte_carlo_zad_3_2_f():
    n = 1000
    U = np.random.uniform(size=n) * 2
    theta = 1 / 2 - 1 / np.pi * (0.93 - 0.46 * np.mean(U ** 2) + 0.07 * np.mean(U ** 4))
    sigma = np.std(1 / (np.pi * (1 + U ** 2)) + 1 / (np.pi * (1 + (2 - U) ** 2)))
    print('Metoda z regrsji liniowej')
    print(f'theta = {theta} +- {sigma * 2 / np.sqrt(n)}')


def ruina_gracza():
    m = 1000000
    n = 100000
    u = 10
    b = 10
    mi = 0.1
    R = []
    for i in range(m):
        Y = []
        S = 0
        R = []
        for i in range(n):
            Y.append(np.random.normal(loc=-mi, scale=1))
            S += Y[i]
            if S > u:
                R.append(1)
            if S < -b:
                R.append(0)
    print(np.mean(R))
    print(2 * np.std(R) / np.sqrt(m))


def ruina_gracza_odwrocenie_dryfu():
    m = 1000000
    n = 100000
    u = 10
    b = 10
    mi = 0.5
    R = []
    for i in range(m):
        Y = []
        S = 0
        for i in range(n):
            Y.append(np.random.normal(loc=mi, scale=1))
            S += Y[i]
            if S > u:
                R.append(np.exp(-2 * mi * S))
                break
            if S < -b:
                R.append(0)
                break
    print(np.mean(R))
    print(2 * np.std(R) / np.sqrt(m))


def SAW():
    n = 100000
    k = 4
    can_all = np.ones(n)
    for i in range(n):

        been_there = np.zeros([2 * k + 3, 2 * k + 3])
        been_there[k + 2, k + 2] = 1
        point = [k + 2, k + 2]
        points = [point.copy()]

        for j in range(k):
            X = np.random.uniform(size=k)
            can = 0
            can_go = []
            if been_there[point[0] + 1, point[1]] == 0:
                can += 1
                can_go.append([point[0] + 1, point[1]])
            if been_there[point[0] - 1, point[1]] == 0:
                can += 1
                can_go.append([point[0] - 1, point[1]])
            if been_there[point[0], point[1] + 1] == 0:
                can += 1
                can_go.append([point[0], point[1] + 1])
            if been_there[point[0], point[1] - 1] == 0:
                can += 1
                can_go.append([point[0], point[1] - 1])
            can_all[i] *= can
            ile = len(can_go)
            for w in range(1, ile + 1):
                if (w - 1) / ile < X[j] < w / ile:
                    point[0] = can_go[w - 1][0]
                    point[1] = can_go[w - 1][1]
                    break

            been_there[point[0], point[1]] = 1
            points.append(point.copy())

    print(np.mean(can_all))
    print(2 * np.std(can_all) / np.sqrt(n))


def MCMC():
    m = 30
    x = m / 2
    n = 1000000
    theta = 1 / 3
    X = []
    for i in range(n):
        u = np.random.uniform()

        if u <= 0.5:
            if np.random.uniform() < (x / (m - x + 1)) * ((1 - theta) / theta):
                x = max(0, x - 1)
            else:
                x = x
        else:
            if np.random.uniform() < ((m - x) / (x + 1)) * (theta / (1 - theta)):
                x = min(m, x + 1)
            else:
                x = x
        X.append(x)
    X_d = []
    Y_d = []
    for w in range(0, m + 1):
        X_d.append(w + 0.5)
        Y_d.append(
            np.math.factorial(m) / (np.math.factorial(m - w) * np.math.factorial(w)) * theta ** w * (1 - theta) ** (
                        m - w))
        print(
            f'{w}: {np.math.factorial(m) / (np.math.factorial(m - w) * np.math.factorial(w)) * theta ** w * (1 - theta) ** (m - w)}')
    plt.hist(X, bins=np.arange(31), density=True)
    plt.scatter(X_d, Y_d, color='red')
    plt.show()


def MCMC_normal():
    x = 0
    n = 1000000
    X = []
    for i in range(n):
        x_prim = np.random.normal()
        if np.random.uniform() < np.exp(-0.5 * (x_prim ** 2 - x ** 2)):
            x = x_prim
        else:
            x = x
        X.append(x)
    plt.hist(X, bins=100, density=True)
    plt.show()


def zad_4_2():
    alpha = -0.9
    eps = np.random.normal(size=1000000)
    x = np.zeros(shape=1000000)
    x[0] = eps[0]
    for i in range(1, len(x)):
        x[i] = alpha * x[i - 1] + eps[i]

    x = np.reshape(x, newshape=(1000, 1000))
    sigma_as = np.sum(np.power(np.mean(x, axis=1) - np.mean(x), 2))
    sigma_norm = np.var(x)

    print(sigma_as, sigma_norm, sigma_norm * (1 + alpha) / (1 - alpha))


def zad_4_1():
    alpha = 0.9
    beta = 0.9

    P = np.array([[1 - alpha, alpha], [beta, 1 - beta]])
    x = np.ones(shape=1000000, dtype='int32')

    for i in range(1, len(x)):
        eps = np.random.uniform()
        if x[i - 1] == 1:
            if eps < P[0, 0]:
                x[i] = 1
            else:
                x[i] = 2
        else:
            if eps < P[1, 1]:
                x[i] = 2
            else:
                x[i] = 1

    x = np.reshape(x, newshape=(1000, 1000))
    sigma_as = np.sum(np.power(np.mean(x, axis=1) - np.mean(x), 2))
    sigma_norm = np.var(x)

    print(sigma_as, sigma_norm, sigma_norm * (2 - alpha - beta) / (alpha + beta))


def Model_AutoLogistyczny():
    d = 5
    alpha = 0.5
    W = np.zeros(shape=(d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                W[i, j] = alpha
            elif i == j + 1 or i == j - 1:
                W[i, j] = 1

    x = np.zeros(shape=(100, d))
    for n in range(100):
        for i in range(d):
            sum = np.dot(W[i, :], x[n - 1, :]) - W[i, i] * (x[n - 1, i] - 1)
            p = 1 / (1 + np.exp(-sum))
            x[n, i] = np.random.choice(a=[1, 0], p=[p, 1 - p])
    print(x[:,0])
    print(np.corrcoef(x, rowvar=False))
    plt.imshow(np.corrcoef(x, rowvar=False), cmap='Blues')
    plt.show()
    plt.hist(x[:,0], density=True)
    plt.show()


Model_AutoLogistyczny()
