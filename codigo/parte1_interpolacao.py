"""TC3 parte 1: interpolação polinomial (exercícios 1 a 4)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metodos import (avalia_newton, coeficientes_newton,
                     diferencas_divididas, interpola,
                     newton_para_potencias)

PASTA_FIG = Path(__file__).resolve().parent.parent / "figuras"
PASTA_FIG.mkdir(exist_ok=True)
np.set_printoptions(suppress=True, linewidth=120)


def mostra_tabela(x, tabela):
    """Imprime a tabela de diferenças divididas como no slide 24."""
    n = len(x)
    print("   x_i  " + "".join(f"{'ordem ' + str(k):>13s}"
                             for k in range(n)))
    for i in range(n):
        print(f"{x[i]:6g}  " + "".join(f"{tabela[i, k]:13.6g}"
                                      for k in range(i + 1)))


def mostra_potencias(a):
    termos = [f"{a[k]:+.6g} x^{k}" for k in range(len(a) - 1, -1, -1)]
    print("p(x) =", " ".join(termos))


def cond_vandermonde(x):
    """Número de condição da matriz de Vandermonde do slide 9."""
    return np.linalg.cond(np.vander(x, increasing=True))


def compara(nome, x, y, pontos, exatos):
    """Interpola, estima nos pontos pedidos e compara com os valores
    conhecidos."""
    c = coeficientes_newton(x, y)
    estimado = avalia_newton(c, x, pontos)
    conferencia = interpola(x, y, pontos)       # função do professor
    erro_nos = np.max(np.abs(avalia_newton(c, x, x) - y))
    print(f"\n{nome}: coeficientes c_k da forma de Newton")
    for k, ck in enumerate(c):
        print(f"  c{k} = {ck: .6e}")
    print(f"  maior |p(x_i) - y_i| nos nós: {erro_nos:.1e}")
    print("  diferença para a função interpola do lab: "
          f"{np.max(np.abs(estimado - conferencia)):.1e}")
    print(f"  {'x':>6s} {'p(x)':>10s} {'conhecido':>10s} {'erro':>10s}"
          f" {'erro rel.':>9s} {'|prod(x-xi)|':>13s}")
    for xp, pe, ex in zip(pontos, estimado, exatos):
        prod = np.prod(np.abs(xp - np.asarray(x)))
        print(f"  {xp:6g} {pe:10.6f} {ex:10.6f} {pe - ex:+10.6f}"
              f" {abs(pe - ex) / ex:9.2%} {prod:13.3e}")
    return c, estimado


# ------------------------------------------------------------- Exercício 1
print("=" * 70)
print("Exercício 1: polinômio de grau <= 3")
x1 = np.array([1.2, 2.1, 3.0, 3.6])
y1 = np.array([0.7, 8.1, 27.7, 45.1])
tab1 = diferencas_divididas(x1, y1)
mostra_tabela(x1, tab1)
c1 = np.diag(tab1)
a1 = newton_para_potencias(c1, x1)
mostra_potencias(a1)
print("p(x_i) =", avalia_newton(c1, x1, x1), " (dados:", y1, ")")

# conferência pelo sistema de Vandermonde (slide 9)
V1 = np.vander(x1, increasing=True)
a1_vander = np.linalg.solve(V1, y1)
print("coeficientes pelo sistema de Vandermonde:", a1_vander)
print("diferença para a forma de Newton expandida: "
      f"{np.max(np.abs(a1 - a1_vander)):.1e}")

# ------------------------------------------------------------- Exercício 2
print("\n" + "=" * 70)
print("Exercício 2: grau <= 5 por diferenças divididas de Newton")
x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y2 = np.array([14.5, 19.5, 30.5, 53.5, 94.5, 159.5])
tab2 = diferencas_divididas(x2, y2)
mostra_tabela(x2, tab2)
c2 = np.diag(tab2)
print("coeficientes c_k:", c2)
a2 = newton_para_potencias(c2, x2)
mostra_potencias(a2)
p45 = avalia_newton(c2, x2, 4.5)
print(f"p(4.5) = {p45:.10f}   exato = 71.375   erro = {p45 - 71.375:.1e}")

# ------------------------------------------------------------- Exercício 3
print("\n" + "=" * 70)
print("Exercício 3: pressão de vapor da água")
T3 = np.array([0, 10, 20, 30, 40, 60, 80, 100], float)
P3 = np.array([0.0061, 0.0123, 0.0234, 0.0424,
               0.0738, 0.1992, 0.4736, 1.0133])
pontos3 = np.array([5.0, 45.0, 95.0])
exatos3 = np.array([0.008721, 0.095848, 0.84528])
c3, est3 = compara("Exercício 3 (grau <= 7)", T3, P3, pontos3, exatos3)
print("  número de condição da matriz de Vandermonde: "
      f"{cond_vandermonde(T3):.1e}")

# ------------------------------------------------------------- Exercício 4
print("\n" + "=" * 70)
print("Exercício 4: propriedade C do titânio")
T4 = np.array([605, 685, 725, 765, 825, 855, 875], float)
C4 = np.array([0.622, 0.655, 0.668, 0.679, 0.730, 0.907, 1.336])
pontos4 = np.array([645.0, 795.0, 845.0])
exatos4 = np.array([0.639, 0.694, 0.812])
c4, est4 = compara("Exercício 4 (grau <= 6)", T4, C4, pontos4, exatos4)
print("  número de condição da matriz de Vandermonde: "
      f"{cond_vandermonde(T4):.1e}")
malha4 = np.linspace(605, 875, 2001)
p_malha4 = avalia_newton(c4, T4, malha4)
i_min = np.argmin(p_malha4)
print(f"  menor valor de p em [605, 875]: {p_malha4[i_min]:.4f}"
      f" em T = {malha4[i_min]:.1f}")

# extra: em vez dos 7 pontos, usar só os 4 nós mais próximos (grau 3)
print("\n  Extra: interpolação só com os 4 nós mais próximos (grau <= 3)")
for xp, ex in zip(pontos4, exatos4):
    perto = np.sort(np.argsort(np.abs(T4 - xp))[:4])
    c_loc = coeficientes_newton(T4[perto], C4[perto])
    pe = avalia_newton(c_loc, T4[perto], xp)
    print(f"  T = {xp:g}: nós {T4[perto].astype(int)} -> p = {pe:.6f},"
          f" erro = {pe - ex:+.6f} ({abs(pe - ex) / ex:.2%})")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.4))

xx = np.linspace(x1[0], x1[-1], 300)
ax[0, 0].plot(xx, avalia_newton(c1, x1, xx), color="C0", label="$p_3(x)$")
ax[0, 0].plot(x1, y1, "o", color="k", label="dados")
ax[0, 0].set_title("Exercício 1")
ax[0, 0].set_xlabel("$x$")
ax[0, 0].set_ylabel("$y$")

xx = np.linspace(x2[0], x2[-1], 300)
ax[0, 1].plot(xx, avalia_newton(c2, x2, xx), color="C0", label="$p(x)$")
ax[0, 1].plot(x2, y2, "o", color="k", label="dados")
ax[0, 1].plot(4.5, p45, "s", mfc="none", ms=8, color="C3",
              label="$p(4{,}5)$")
ax[0, 1].set_title("Exercício 2")
ax[0, 1].set_xlabel("$x$")
ax[0, 1].set_ylabel("$f(x)$")

xx = np.linspace(T3[0], T3[-1], 500)
ax[1, 0].plot(xx, avalia_newton(c3, T3, xx), color="C0", label="$p_7(T)$")
ax[1, 0].plot(T3, P3, "o", color="k", label="dados")
ax[1, 0].plot(pontos3, exatos3, "x", ms=8, mew=2, color="C3",
              label="valores conhecidos")
ax[1, 0].set_title("Exercício 3: pressão de vapor")
ax[1, 0].set_xlabel("$T$ (°C)")
ax[1, 0].set_ylabel("$P$ (bar)")

ax[1, 1].plot(malha4, p_malha4, color="C0", label="$p_6(T)$")
ax[1, 1].plot(T4, C4, "o", color="k", label="dados")
ax[1, 1].plot(pontos4, exatos4, "x", ms=8, mew=2, color="C3",
              label="valores conhecidos")
ax[1, 1].set_title("Exercício 4: titânio")
ax[1, 1].set_xlabel("$T$")
ax[1, 1].set_ylabel("$C$")

for eixo in ax.flat:
    eixo.grid(alpha=.3)
    eixo.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PASTA_FIG / "parte1_interpolacao.png", dpi=300)
print("\nfigura salva em figuras/parte1_interpolacao.png")
