"""Funções do TC3 de Cálculo Numérico: interpolação e mínimos quadrados.

As funções de mínimos quadrados (equacoes_normais, resolve, avalia e
soma_quadrados) são as que o grupo escreveu no laboratório de 13/10.
A função interpola é a que veio pronta no notebook desse laboratório.
"""
import numpy as np


# ------------------------------------------------------------ interpolação

def diferencas_divididas(x, y):
    """Tabela de diferenças divididas (slides 17 a 24 de interpolação).

    tabela[i, k] = f[x_(i-k), ..., x_i]. A coluna 0 são os dados e os
    coeficientes da forma de Newton ficam na diagonal.
    """
    x = np.asarray(x, float)
    n = len(x)
    tabela = np.zeros((n, n))
    tabela[:, 0] = y                       # ordem 0: os dados
    for k in range(1, n):                  # ordem k
        for i in range(k, n):
            # (vizinha à esquerda - a de cima dela) / (x_i - x_(i-k))
            tabela[i, k] = ((tabela[i, k - 1] - tabela[i - 1, k - 1])
                            / (x[i] - x[i - k]))
    return tabela


def coeficientes_newton(x, y):
    """c_k = f[x_0, ..., x_k], a diagonal da tabela."""
    return np.diag(diferencas_divididas(x, y)).copy()


def avalia_newton(c, x_nos, x):
    """Avalia p(x) = c0 + c1 (x - x0) + ... + cn (x - x0)...(x - x_(n-1)).

    Começa pelo último coeficiente e vai multiplicando por (x - x_k) e
    somando c_k, como no esquema de Horner.
    """
    x = np.asarray(x, float)
    p = np.full_like(x, c[-1])
    for k in range(len(c) - 2, -1, -1):
        p = p * (x - x_nos[k]) + c[k]
    return p


def newton_para_potencias(c, x_nos):
    """Coeficientes a0, a1, ..., an de p(x) = a0 + a1 x + ... + an x^n."""
    P = np.polynomial.polynomial
    a = np.array([c[-1]])
    for k in range(len(c) - 2, -1, -1):
        a = P.polymul(a, [-x_nos[k], 1.0])     # vezes (x - x_k)
        a = P.polyadd(a, [c[k]])               # mais c_k
    return a


def interpola(xn, yn, x):
    """Avalia em x o polinômio que interpola os pontos (xn, yn).

    Função pronta do notebook do laboratório de 13/10. É o mesmo
    polinômio de Lagrange/Newton, só que calculado pela fórmula
    baricêntrica.
    """
    xn = np.asarray(xn, float)
    yn = np.asarray(yn, float)
    w = np.array([1.0 / np.prod(xn[j] - np.delete(xn, j))
                  for j in range(len(xn))])
    x = np.atleast_1d(np.asarray(x, float))
    p = np.empty_like(x)
    for k, xk in enumerate(x):
        dif = xk - xn
        if np.any(dif == 0):                      # x coincide com um nó
            p[k] = yn[np.argmin(np.abs(dif))]
        else:
            q = w / dif
            p[k] = (q @ yn) / q.sum()
    return p


# ------------------------------------------------------- mínimos quadrados

def equacoes_normais(t, y, grau):
    """Monta A e b das equações normais para o modelo
        phi(t) = a0 + a1 t + ... + a_grau t^grau
    com  A[j, k] = soma_i t_i^(j+k)   e   b[j] = soma_i t_i^j y_i.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = grau + 1                # número de coeficientes a0, ..., a_grau
    A = np.zeros((n, n))
    b = np.zeros(n)
    for j in range(n):          # a linha j vem de dE/da_j = 0
        for k in range(n):
            A[j, k] = np.sum(t**(j + k))
        b[j] = np.sum(t**j * y)
    return A, b


def resolve(A, b):
    """Resolve A a = b por eliminação de Gauss com pivoteamento parcial
    (aula de sistemas lineares), seguida de retrossubstituição."""
    A = np.array(A, dtype=float)    # cópias, para não alterar A e b de fora
    b = np.array(b, dtype=float)
    n = len(b)

    # eliminação: zera os elementos abaixo da diagonal, coluna por coluna
    for k in range(n - 1):
        # pivoteamento parcial: traz para a linha k o maior |A[i, k]|,
        # com i >= k
        p = k + np.argmax(np.abs(A[k:, k]))
        if A[p, k] == 0:
            raise ValueError("matriz singular")
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]        # multiplicador
            A[i, k:] = A[i, k:] - m * A[k, k:]
            b[i] = b[i] - m * b[k]
    if A[n - 1, n - 1] == 0:
        raise ValueError("matriz singular")

    # retrossubstituição
    a = np.zeros(n)
    for i in range(n - 1, -1, -1):
        a[i] = (b[i] - A[i, i + 1:] @ a[i + 1:]) / A[i, i]
    return a


def avalia(coef, t):
    """Avalia phi(t) = coef[0] + coef[1] t + coef[2] t^2 + ..."""
    t = np.asarray(t, float)
    return sum(c * t**k for k, c in enumerate(coef))


def soma_quadrados(coef, t, y):
    """E = soma dos quadrados dos resíduos r_i = y_i - phi(t_i)."""
    r = np.asarray(y, float) - avalia(coef, t)
    return np.sum(r**2)


def ajusta(x, y, grau):
    """Polinômio de mínimos quadrados: monta e resolve as equações
    normais."""
    A, b = equacoes_normais(x, y, grau)
    return resolve(A, b)
