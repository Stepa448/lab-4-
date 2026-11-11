"""Лабораторная работа № 4. Вариант 11.
Последовательный, двоичный и интерполяционный поиск.
Данные: кластеризованное распределение ключей (К), n = 10^3, 10^4, 10^5,
k = 500 успешных запросов. Доп. задание Б4: границы диапазона и число вхождений.

Автор: Колесный Степан, группа БИ2-1.
Запуск: python lab4_search.py   (графики строятся, если установлен matplotlib)
"""
import bisect
import math
import os
import platform
import random
import time

SIZES = [10 ** 3, 10 ** 4, 10 ** 5]  # размеры массивов по варианту
K = 500                                # число запросов по варианту


# ---------------------------------------------------------------- алгоритмы поиска
def linear_search(a, q):
    """Последовательный поиск. Возвращает (индекс или -1, число сравнений)."""
    for i, value in enumerate(a):
        if value == q:
            return i, i + 1
    return -1, len(a)


def binary_search(a, q):
    """Двоичный поиск в упорядоченном массиве."""
    lo, hi, cmps = 0, len(a) - 1, 0
    while lo <= hi:
        mid = lo + (hi - lo) // 2          # безопасная форма середины отрезка
        cmps += 1
        if a[mid] == q:
            return mid, cmps
        if a[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, cmps


def interpolation_search(a, q):
    """Интерполяционный поиск в упорядоченном массиве."""
    lo, hi, cmps = 0, len(a) - 1, 0
    while lo <= hi and a[lo] <= q <= a[hi]:
        if a[lo] == a[hi]:                 # все элементы отрезка равны: деления на 0 быть не должно
            cmps += 1
            return (lo, cmps) if a[lo] == q else (-1, cmps)
        mid = lo + (q - a[lo]) * (hi - lo) // (a[hi] - a[lo])  # ожидаемая позиция ключа
        cmps += 1
        if a[mid] == q:
            return mid, cmps
        if a[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, cmps


# ---------------------------------------------------------------- данные и замеры
def make_clustered(n, seed=42):
    """Распределение К: 90 % ключей из [0; n], 10 % из [10^6; 10^7]. Ключи различны, массив упорядочен."""
    rng = random.Random(seed)
    n_low = int(0.9 * n)
    low = rng.sample(range(0, n + 1), n_low)               # 90 % - плотный кластер
    high = rng.sample(range(10 ** 6, 10 ** 7 + 1), n - n_low)  # 10 % - далёкие «выбросы»
    return sorted(low + high)


def benchmark(search_func, data, queries):
    """Среднее время одного запроса (с) и среднее число сравнений."""
    start = time.perf_counter()
    total = 0
    for q in queries:
        _, cmps = search_func(data, q)
        total += cmps
    return (time.perf_counter() - start) / len(queries), total / len(queries)


# ---------------------------------------------------------------- доп. задание Б4
def lower_bound(a, q):
    """Индекс первого элемента >= q и число сравнений. O(log n)."""
    lo, hi, cmps = 0, len(a), 0
    while lo < hi:
        mid = lo + (hi - lo) // 2
        cmps += 1
        if a[mid] < q:
            lo = mid + 1
        else:
            hi = mid
    return lo, cmps


def upper_bound(a, q):
    """Индекс первого элемента > q и число сравнений. O(log n)."""
    lo, hi, cmps = 0, len(a), 0
    while lo < hi:
        mid = lo + (hi - lo) // 2
        cmps += 1
        if a[mid] <= q:
            lo = mid + 1
        else:
            hi = mid
    return lo, cmps


def equal_range(a, q):
    """Границы вхождений ключа: (первый индекс, последний индекс, количество, сравнения)."""
    left, c1 = lower_bound(a, q)
    right, c2 = upper_bound(a, q)
    if left == right:                      # ключа в массиве нет
        return -1, -1, 0, c1 + c2
    return left, right - 1, right - left, c1 + c2


def test_equal_range(n=10 ** 5, trials=2000, seed=7):
    """Проверка Б4 на массиве с повторами: сравнение с bisect и с полным подсчётом."""
    rng = random.Random(seed)
    a = sorted(rng.randint(0, n // 10) for _ in range(n))   # в среднем ~10 повторов на ключ
    ok, cmps_sum = True, 0
    for _ in range(trials):
        q = rng.randint(-5, n // 10 + 5)                    # есть и существующие, и отсутствующие ключи
        first, last, cnt, cmps = equal_range(a, q)
        ref_l, ref_r = bisect.bisect_left(a, q), bisect.bisect_right(a, q)
        ok &= (cnt == ref_r - ref_l) and (cnt == a.count(q))
        ok &= (first == (ref_l if cnt else -1)) and (last == (ref_r - 1 if cnt else -1))
        cmps_sum += cmps
    return ok, cmps_sum / trials, math.log2(n)


# ---------------------------------------------------------------- запуск
def system_info():
    """Характеристики вычислительной системы для отчёта."""
    return (f"ОС: {platform.platform()}\nПроцессор: {platform.processor() or platform.machine()}\n"
            f"Логических ядер: {os.cpu_count()}\nPython: {platform.python_version()}")


def plot(results):
    """Графики времени и числа сравнений (логарифмические оси). Сохраняет graphs.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - графики пропущены (pip install matplotlib)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ("Последовательный", "Двоичный", "Интерполяционный"):
        ax[0].plot(SIZES, [results[n][name][0] * 1e6 for n in SIZES], marker="o", label=name)
        ax[1].plot(SIZES, [results[n][name][1] for n in SIZES], marker="o", label=name)
    ax[1].plot(SIZES, [(n + 1) / 2 for n in SIZES], "k--", lw=0.8, label="теория (n+1)/2")
    ax[1].plot(SIZES, [math.log2(n) for n in SIZES], "k:", lw=0.8, label="теория log2 n")
    for a_, title, yl in zip(ax, ("Среднее время одного запроса", "Среднее число сравнений"), ("мкс", "сравнений")):
        a_.set(xscale="log", yscale="log", xlabel="n", ylabel=yl, title=title)
        a_.grid(True, which="both", alpha=0.3)
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("graphs.png", dpi=150)
    print("Графики сохранены в graphs.png")


if __name__ == "__main__":
    print("=== Характеристики системы ===")
    print(system_info())
    algorithms = {"Последовательный": linear_search, "Двоичный": binary_search,
                  "Интерполяционный": interpolation_search}
    rng = random.Random(1)
    results = {}
    for n in SIZES:
        data = make_clustered(n)
        queries = [rng.choice(data) for _ in range(K)]      # успешные запросы
        results[n] = {name: benchmark(f, data, queries) for name, f in algorithms.items()}

    print("\n=== Среднее время одного запроса, мкс ===")
    print(f"{'n':>8}" + "".join(f"{name:>20}" for name in algorithms))
    for n in SIZES:
        print(f"{n:>8}" + "".join(f"{results[n][name][0] * 1e6:>20.2f}" for name in algorithms))

    print("\n=== Среднее число сравнений ===")
    print(f"{'n':>8}{'послед.':>12}{'(n+1)/2':>12}{'двоичн.':>10}{'log2 n':>10}{'интерп.':>10}{'log2log2 n':>12}")
    for n in SIZES:
        r = results[n]
        print(f"{n:>8}{r['Последовательный'][1]:>12.1f}{(n + 1) / 2:>12.1f}{r['Двоичный'][1]:>10.1f}"
              f"{math.log2(n):>10.1f}{r['Интерполяционный'][1]:>10.1f}{math.log2(math.log2(n)):>12.1f}")

    print("\n=== Б4: границы диапазона и число вхождений ===")
    demo = [1, 2, 2, 2, 3, 5, 5, 8]
    print("Массив:", demo)
    for q in (2, 5, 4):
        print(f"  q = {q}: (первый, последний, количество, сравнений) =", equal_range(demo, q))
    ok, avg_cmps, lg = test_equal_range()
    print(f"Проверка на 2000 запросах (n = 10^5, с повторами): совпадает с bisect и подсчётом: {ok}")
    print(f"Среднее число сравнений equal_range: {avg_cmps:.1f} (2*log2 n = {2 * lg:.1f}), линейный подсчёт: ~10^5")
    plot(results)
