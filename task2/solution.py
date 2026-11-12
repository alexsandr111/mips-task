"""Практическая работа №2. Вариант 11. Python 3.10+."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def f(x, y):
    return 5*x + 3*math.cos(y + 2.6)


def euler(x, y, h):
    return y + h*f(x, y)


def euler_cauchy(x, y, h):
    predictor = y + h*f(x, y)
    return y + h*(f(x, y) + f(x+h, predictor))/2


def runge_kutta(x, y, h):
    k1 = h*f(x, y)
    k2 = h*f(x+h/2, y+k1/2)
    k3 = h*f(x+h/2, y+k2/2)
    k4 = h*f(x+h, y+k3)
    return y + (k1+2*k2+2*k3+k4)/6


def solve_cauchy():
    """Задание 1: y'=5x+3cos(y+2.6), y(1)=1.5, h=0.2, x in [1,3]."""
    methods = (euler, euler_cauchy, runge_kutta)
    ys = [1.5]*3
    rows = []
    for i in range(11):
        x = 1 + 0.2*i
        rows.append(dict(i=i, x=x, euler=ys[0],
                         euler_cauchy=ys[1], rk4=ys[2]))
        if i < 10:
            ys = [method(x, y, 0.2) for method, y in zip(methods, ys)]
    return rows


def gauss(matrix, rhs):
    """Метод Гаусса с частичным выбором главного элемента."""
    a = [r.copy() for r in matrix]
    b = rhs.copy()
    n = len(b)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ValueError("Вырожденная система")
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        for i in range(col+1, n):
            m = a[i][col]/a[col][col]
            for j in range(col, n):
                a[i][j] -= m*a[col][j]
            b[i] -= m*b[col]
    y = [0.0]*n
    for i in range(n-1, -1, -1):
        y[i] = (b[i]-sum(a[i][j]*y[j] for j in range(i+1,n)))/a[i][i]
    return y


def boundary_matrix():
    """Односторонние конечные разности из методички на узлах 0.8..1.1."""
    a, h, n = 0.8, 0.1, 3
    xs = [a+i*h for i in range(n+1)]
    matrix = [[0.0]*(n+1) for _ in range(n+1)]
    rhs = [0.0]*(n+1)
    for i in range(n-1):
        p, q, r = -3/xs[i], 5, 9
        matrix[i][i] = 1-h*p+h*h*q
        matrix[i][i+1] = h*p-2
        matrix[i][i+2] = 1.0
        rhs[i] = h*h*r
    # y'(0.8)=1.6
    matrix[n-1][0], matrix[n-1][1], rhs[n-1] = -1, 1, h*1.6
    # 2y(1.1)+3y'(1.1)=4.8
    matrix[n][n-1], matrix[n][n], rhs[n] = -3, 3+2*h, h*4.8
    return xs, matrix, rhs


def solve_boundary():
    """Задание 2: y''-3y'/x+5y=9, шаг 0.1."""
    xs, matrix, rhs = boundary_matrix()
    ys = gauss(matrix, rhs)
    residual = max(abs(sum(a*v for a,v in zip(row,ys))-b)
                   for row,b in zip(matrix,rhs))
    return [dict(i=i,x=x,y=y) for i,(x,y) in enumerate(zip(xs,ys))], residual


def write_csv(filename, rows):
    with (OUT/filename).open("w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_results(first, second):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib отсутствует: CSV созданы, PNG не обновлены.")
        return
    fig, ax = plt.subplots(figsize=(9,5))
    for key, label in (("euler","Euler"),("euler_cauchy","Euler-Cauchy"),
                       ("rk4","Runge-Kutta 4")):
        ax.plot([r["x"] for r in first], [r[key] for r in first],
                marker="o", label=label)
    ax.set(xlabel="x", ylabel="y(x)", title="Task 1: Cauchy problem, variant 11")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/"plot_task1.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot([r["x"] for r in second], [r["y"] for r in second],
            marker="o", label="Finite differences")
    ax.set(xlabel="x", ylabel="y(x)", title="Task 2: boundary-value problem")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/"plot_task2.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    first = solve_cauchy()
    second, residual = solve_boundary()
    write_csv("task1.csv", first)
    write_csv("task2.csv", second)
    plot_results(first, second)
    print("Задача Коши:")
    print("  x       Эйлер    Эйлер-Коши        RK4")
    for row in first:
        print(f'{row["x"]:5.1f} {row["euler"]:12.6f} '
              f'{row["euler_cauchy"]:12.6f} {row["rk4"]:12.6f}')
    print("Краевая задача:")
    for row in second:
        print(f'y({row["x"]:.1f}) = {row["y"]:.9f}')
    print(f'Максимальная невязка СЛАУ: {residual:.3e}')
    summary = ("ПЗ №2, вариант 11\n"
               + "Задача Коши: y(3) (Эйлер / Эйлер-Коши / RK4): "
               + " / ".join(f'{first[-1][key]:.10f}' for key in
                            ("euler","euler_cauchy","rk4")) + "\n"
               + "Краевая задача: \n"
               + "\n".join(f'y({r["x"]:.1f}) = {r["y"]:.10f}' for r in second)
               + f"\nНевязка: {residual:.3e}\n")
    (OUT/"summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
