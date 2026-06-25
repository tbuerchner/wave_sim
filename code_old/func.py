import numpy as np

# Gauss-Lobatto points and weights taken from here:
# https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb
def lgP(n, xi):
    if n == 0:
        return np.ones(xi.size)

    elif n == 1:
        return xi

    else:
        fP = np.ones(xi.size)
        sP = xi.copy()
        nP = np.empty(xi.size)
        for i in range(2, n + 1):
            nP = ((2 * i - 1) * xi * sP - (i - 1) * fP) / i
            fP = sP
            sP = nP

        return nP

def GLL(n, epsilon=1e-15):
    if n < 2:
        print('Error: n must be larger than 1')

    else:
        x = np.empty(n)
        w = np.empty(n)

        x[0] = -1
        x[n - 1] = 1
        w[0] = w[0] = 2.0 / ((n * (n - 1)))
        w[n - 1] = w[0]

        n_2 = n // 2

        dLgP = lambda n, xi: n * (lgP(n - 1, xi) - xi * lgP(n, xi)) / (1 - xi ** 2)
        d2LgP = lambda n, xi: (2 * xi * dLgP(n, xi) - n * (n + 1) * lgP(n, xi)) / (1 - xi ** 2)
        d3LgP = lambda n, xi: (4 * xi * d2LgP(n, xi) - (n * (n + 1) - 2) * dLgP(n, xi)) / (1 - xi ** 2)

        for i in range(1, n_2):
            xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) * \
                 np.cos((4 * i + 1) * np.pi / (4 * (n - 1) + 1))

            error = 1.0

            while error > epsilon:
                y = dLgP(n - 1, xi)
                y1 = d2LgP(n - 1, xi)
                y2 = d3LgP(n - 1, xi)

                dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)

                xi -= dx
                error = abs(dx)

            x[i] = -xi
            x[n - i - 1] = xi

            w[i] = 2 / (n * (n - 1) * lgP(n - 1, x[i]) ** 2)
            w[n - i - 1] = w[i]

        if n % 2 != 0:
            x[n_2] = 0
            w[n_2] = 2.0 / ((n * (n - 1)) * lgP(n - 1, np.array(x[n_2])) ** 2)

    return np.array(x), np.array(w)

def get_element_index(i_element, j_element, n_elements):

    return j_element * n_elements[0] + i_element