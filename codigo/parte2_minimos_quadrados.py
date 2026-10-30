"""TC3 parte 2: ajuste por mínimos quadrados (exercícios 2 a 6)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metodos import ajusta, avalia, equacoes_normais, soma_quadrados

PASTA_FIG = Path(__file__).resolve().parent.parent / "figuras"
PASTA_FIG.mkdir(exist_ok=True)
np.set_printoptions(suppress=True, linewidth=120)


def formata(v):
    """Número com 6 casas, ou em notação científica se for quase zero."""
    return f"{v: .1e}" if v != 0 and abs(v) < 1e-4 else f"{v: .6f}"


def resultado(nome, x, y, grau):
    """Ajusta, imprime A, b, coeficientes, E e confere com np.polyfit."""
    A, b = equacoes_normais(x, y, grau)
    coef = ajusta(x, y, grau)
    E = soma_quadrados(coef, x, y)
    # o polyfit devolve os coeficientes do maior grau para o menor
    conferencia = np.polyfit(x, y, grau)[::-1]
    print(f"\n{nome}")
    print("A =\n", A)
    print("b =", b)
    print(f"  número de condição de A: {np.linalg.cond(A):.1e}")
    for k, a in enumerate(coef):
        print(f"  a{k} = {formata(a)}")
    print(f"  E = soma dos quadrados dos resíduos = {formata(E)}")
    dif = np.max(np.abs(coef - conferencia))
    print(f"  diferença para np.polyfit: {dif:.1e}")
    return coef, E


def tabela_residuos(x, y, *coefs):
    cab = f"  {'x':>6s} {'y':>8s}"
    for c in coefs:
        cab += f" {'p' + str(len(c) - 1) + '(x)':>9s} {'r':>8s}"
    print(cab)
    for xi, yi in zip(x, y):
        linha = f"  {xi:6g} {yi:8g}"
        for c in coefs:
            p = avalia(c, xi)
            linha += f" {p:9.4f} {yi - p:+8.4f}"
        print(linha)


# ------------------------------------------------------------- Exercício 2
print("=" * 70)
print("Exercício 2")
x2 = np.array([1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8])
y2 = np.array([5.5, 7.3, 14.1, 26.3, 48.2, 80.1, 111.6])
c2a, E2a = resultado("(a) grau 2", x2, y2, 2)
c2b, E2b = resultado("(b) grau 3", x2, y2, 3)
tabela_residuos(x2, y2, c2a, c2b)
print(f"  E caiu de {E2a:.4f} para {E2b:.4f} ({E2a / E2b:.1f} vezes menor)")

# ------------------------------------------------------------- Exercício 3
print("\n" + "=" * 70)
print("Exercício 3: dados gerados por f(x) = x^2 + 3x + 7")
x3 = np.array([0.0, 1.0, 2.0, 3.0])
y3 = x3**2 + 3 * x3 + 7
c3_2, E3_2 = resultado("grau 2", x3, y3, 2)
c3_3, E3_3 = resultado("grau 3", x3, y3, 3)
print(f"  {'x':>3s} {'f(x)':>6s} {'p2(x)':>14s} {'p3(x)':>14s}")
for xi, yi in zip(x3, y3):
    print(f"  {xi:3g} {yi:6g} {avalia(c3_2, xi):14.10f}"
          f" {avalia(c3_3, xi):14.10f}")
for nome, c in [("p2", c3_2), ("p3", c3_3)]:
    dif = np.max(np.abs(avalia(c, x3) - y3))
    print(f"  maior |{nome} - f| nos pontos: {dif:.1e}")

# ------------------------------------------------------------- Exercício 4
print("\n" + "=" * 70)
print("Exercício 4: f(x) = sen(pi x) em 5 pontos, grau 3")
x4 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
y4 = np.sin(np.pi * x4)
print("y =", y4)
c4, E4 = resultado("grau 3", x4, y4, 3)
print(f"  8/3 = {8 / 3:.6f}")
print("  p3 nos pontos:", avalia(c4, x4))
malha4 = np.linspace(-1, 1, 2001)
erro4 = np.abs(np.sin(np.pi * malha4) - avalia(c4, malha4))
i_max = np.argmax(erro4)
print(f"  maior |sen(pi x) - p3(x)| em [-1, 1]: {erro4[i_max]:.4f}"
      f" em x = {malha4[i_max]:+.3f}")
taylor = np.pi * malha4 - np.pi**3 * malha4**3 / 6
print("  maior erro do polinômio de Taylor de grau 3 em [-1, 1]: "
      f"{np.max(np.abs(np.sin(np.pi * malha4) - taylor)):.3f}")

# ------------------------------------------------------------- Exercício 5
print("\n" + "=" * 70)
print("Exercício 5")
x5 = np.array([1.2, 3.0, 4.5, 5.6, 6.9, 8.1])
y5 = np.array([2.3, 3.3, 5.6, 15.4, 38.2, 59.1])
c5, E5 = resultado("grau 2", x5, y5, 2)
tabela_residuos(x5, y5, c5)
print(f"  f(6.0) aproximado por p2(6.0) = {avalia(c5, 6.0):.4f}")
x_v = -c5[1] / (2 * c5[2])                # vértice da parábola
print(f"  vértice da parábola: x = {x_v:.4f}, p2 = {avalia(c5, x_v):.4f}")

# ------------------------------------------------------------- Exercício 6
print("\n" + "=" * 70)
print("Exercício 6")
x6 = np.array([1.0, 2.3, 3.8, 4.9, 6.1, 7.2])
y6 = np.array([0.1, 1.5, 7.4, 15.2, 26.2, 27.9])
c6, E6 = resultado("grau 2", x6, y6, 2)
tabela_residuos(x6, y6, c6)

# ------------------------------------------------------------------ figura
fig = plt.figure(figsize=(7.6, 4.4))
grade = fig.add_gridspec(2, 6)
ax = [fig.add_subplot(grade[0, 0:2]), fig.add_subplot(grade[0, 2:4]),
      fig.add_subplot(grade[0, 4:6]), fig.add_subplot(grade[1, 1:3]),
      fig.add_subplot(grade[1, 3:5])]

xx = np.linspace(x2[0], x2[-1], 300)
ax[0].plot(xx, avalia(c2a, xx), color="C0", label="grau 2")
ax[0].plot(xx, avalia(c2b, xx), "--", color="C1", label="grau 3")
ax[0].plot(x2, y2, "o", color="k", label="dados")
ax[0].set_title("Exercício 2")

xx = np.linspace(-0.2, 3.2, 300)
ax[1].plot(xx, xx**2 + 3 * xx + 7, color="0.6", lw=5, alpha=.5,
           label="$x^2+3x+7$")
ax[1].plot(xx, avalia(c3_2, xx), color="C0", label="grau 2")
ax[1].plot(xx, avalia(c3_3, xx), "--", color="C1", label="grau 3")
ax[1].plot(x3, y3, "o", color="k", label="dados")
ax[1].set_title("Exercício 3")

ax[2].plot(malha4, np.sin(np.pi * malha4), color="0.6", lw=5, alpha=.5,
           label=r"sen$(\pi x)$")
ax[2].plot(malha4, avalia(c4, malha4), color="C0", label="grau 3")
ax[2].plot(x4, y4, "o", color="k", label="pontos")
ax[2].set_title("Exercício 4")

xx = np.linspace(x5[0], x5[-1], 300)
ax[3].plot(xx, avalia(c5, xx), color="C0", label="grau 2")
ax[3].plot(x5, y5, "o", color="k", label="dados")
ax[3].plot(6.0, avalia(c5, 6.0), "s", mfc="none", ms=8, color="C3",
           label="$p_2(6{,}0)$")
ax[3].set_title("Exercício 5")

xx = np.linspace(x6[0], x6[-1], 300)
ax[4].plot(xx, avalia(c6, xx), color="C0", label="grau 2")
ax[4].plot(x6, y6, "o", color="k", label="dados")
ax[4].set_title("Exercício 6")

for eixo in ax:
    eixo.set_xlabel("$x$")
    eixo.set_ylabel("$y$")
    eixo.grid(alpha=.3)
    eixo.legend(fontsize=7)
plt.tight_layout()
plt.savefig(PASTA_FIG / "parte2_minimos_quadrados.png", dpi=300)
print("\nfigura salva em figuras/parte2_minimos_quadrados.png")
