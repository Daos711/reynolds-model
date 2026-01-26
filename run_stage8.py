#!/usr/bin/env python3
"""
Этап 8: Параметрические исследования и оптимизация текстуры.

БЛОК 1: Сравнение A/B/C/D (база)
БЛОК 2: Sweep по c (A и C)
БЛОК 3: Оптимизация текстуры (Fn × h_star × паттерн × сектор)
БЛОК 4: Pareto + топ-5
БЛОК 5: Робастность топ-5 (c + μ + K/C)
БЛОК 6: Итоговые графики/таблицы
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

from bearing_solver import (
    BearingConfig,
    SmoothFilmModel,
    TexturedFilmModel,
    TextureParams,
    RoughnessParams,
    solve_reynolds,
    compute_forces,
    compute_stage2,
    compute_roughness_fields,
    find_equilibrium,
    compute_dynamic_coefficients,
    analyze_stability,
    validate_texture,
)


# ============================================================================
# КОНФИГУРАЦИЯ ИССЛЕДОВАНИЯ
# ============================================================================

# Базовые параметры
BASE_R = 0.050          # радиус, м
BASE_L = 0.050          # длина, м
BASE_C = 50e-6          # зазор, м
BASE_N_RPM = 3000       # об/мин
BASE_W_EXT = 50e3       # Н
BASE_MASS = 30.0        # кг

# Вязкость по температуре (VG100)
MU_BY_TEMP = {
    40: 0.098,
    50: 0.057,
    60: 0.037,
    70: 0.025,
}
BASE_TEMP = 50
BASE_MU = MU_BY_TEMP[BASE_TEMP]

# Шероховатость
ROUGHNESS_PARAMS = RoughnessParams(
    Ra_shaft=0.63e-6,
    Ra_out=1.25e-6,
    Ra_cell=0.63e-6,
)

# Параметры текстуры для режима C/D (ГОСТ базовый)
GOST_TEXTURE = TextureParams(
    a=0.5e-3,
    b=0.5e-3,
    h_star=0.10,
    Fn=0.15,
    pattern='phyllotaxis',
)

# Sweep по зазору
C_VALUES = [30e-6, 40e-6, 50e-6, 63e-6, 80e-6, 100e-6, 125e-6, 160e-6, 180e-6]

# Sweep по вязкости (температуры)
TEMP_VALUES = [40, 50, 60, 70]

# Сетка оптимизации текстуры
FN_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50]
H_STAR_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40]
PATTERN_VALUES = ['phyllotaxis', 'regular', 'spiral']
SECTOR_VALUES = [
    (0, 2*np.pi),              # full
    (np.pi/2, 3*np.pi/2),      # 90-270° (нагруженная зона)
]


OUT_DIR = Path("results/stage8")


@dataclass
class CaseResult:
    """Результат расчёта одного кейса."""
    name: str
    epsilon: float
    phi0_deg: float
    W: float
    p_max: float
    h_min: float
    Q: float
    f: float           # коэффициент трения
    P_loss: float      # потери мощности, Вт
    Kxx: Optional[float] = None
    Kyy: Optional[float] = None
    Cxx: Optional[float] = None
    Cyy: Optional[float] = None
    margin: Optional[float] = None
    whirl_ratio: Optional[float] = None
    is_stable: Optional[bool] = None
    frac_lambda_lt_1: float = 0.0
    is_valid: bool = True
    c: float = BASE_C
    mu: float = BASE_MU
    Fn: float = 0.0
    h_star: float = 0.0
    pattern: str = ''


def create_config(c=BASE_C, mu=BASE_MU, epsilon=0.5, phi0=np.pi/2,
                  n_phi=180, n_z=50) -> BearingConfig:
    """Создать конфигурацию подшипника."""
    return BearingConfig(
        R=BASE_R,
        L=BASE_L,
        c=c,
        epsilon=epsilon,
        phi0=phi0,
        n_rpm=BASE_N_RPM,
        mu=mu,
        n_phi=n_phi,
        n_z=n_z,
    )


def smooth_factory(cfg: BearingConfig):
    return SmoothFilmModel(cfg)


def texture_factory_gost(cfg: BearingConfig):
    return TexturedFilmModel(cfg, GOST_TEXTURE)


def create_texture_factory(tex_params: TextureParams):
    def factory(cfg: BearingConfig):
        return TexturedFilmModel(cfg, tex_params)
    return factory


def solve_with_roughness(config, film_model, roughness_params, texture_mask=None):
    """Решить с шероховатостью."""
    phi, Z, _, _ = config.create_grid()
    H = film_model.H(phi, Z)
    roughness_result = compute_roughness_fields(
        H, phi, Z, config.c, roughness_params, texture_mask
    )
    reynolds_result = solve_reynolds(
        config, film_model,
        phi_x=roughness_result.phi_x,
        phi_z=roughness_result.phi_z,
        sigma_star=roughness_result.sigma_star,
        lambda_field=roughness_result.lambda_field,
        frac_lambda_lt_1=roughness_result.frac_lambda_lt_1,
    )
    return reynolds_result, roughness_result


def compute_case(
    name: str,
    base_config: BearingConfig,
    film_model_factory: Callable,
    W_ext: float = BASE_W_EXT,
    use_roughness: bool = False,
    roughness_params: Optional[RoughnessParams] = None,
    compute_dynamics: bool = False,
    tex_params: Optional[TextureParams] = None,
) -> CaseResult:
    """
    Полный расчёт одного кейса: равновесие → статика → (опц.) динамика.
    """
    # Этап 3: Равновесие
    try:
        eq = find_equilibrium(
            base_config, W_ext=W_ext, load_angle=-np.pi/2,
            verbose=False, film_model_factory=film_model_factory,
            coarse_grid=(90, 25), fine_grid=(180, 50)
        )
    except Exception as e:
        # Если не сходится — возвращаем невалидный результат
        print(f"  [WARN] find_equilibrium failed for {name}: {type(e).__name__}: {e}")
        return CaseResult(
            name=name, epsilon=0.99, phi0_deg=0, W=0, p_max=0, h_min=0,
            Q=0, f=0, P_loss=0, is_valid=False,
            c=base_config.c, mu=base_config.mu,
            Fn=tex_params.Fn if tex_params else 0,
            h_star=tex_params.h_star if tex_params and tex_params.h_star else 0,
            pattern=tex_params.pattern if tex_params else '',
        )

    # Конфиг для равновесной позиции
    eq_config = BearingConfig(
        R=base_config.R, L=base_config.L, c=base_config.c,
        epsilon=eq.epsilon, phi0=eq.phi0,
        n_rpm=base_config.n_rpm, mu=base_config.mu,
        n_phi=180, n_z=50
    )

    film_model = film_model_factory(eq_config)

    # Проверка валидности текстуры
    is_valid = True
    if hasattr(film_model, 'is_valid'):
        is_valid = film_model.is_valid

    # Этап 2: Решение Рейнольдса
    frac_lt_1 = 0.0
    if use_roughness and roughness_params:
        phi, Z, _, _ = eq_config.create_grid()
        texture_mask = None
        if hasattr(film_model, 'texture_mask'):
            texture_mask = film_model.texture_mask(phi, Z)
        reynolds, rough = solve_with_roughness(
            eq_config, film_model, roughness_params, texture_mask
        )
        frac_lt_1 = rough.frac_lambda_lt_1
    else:
        reynolds = solve_reynolds(eq_config, film_model)

    # Статика
    stage2 = compute_stage2(reynolds, eq_config)
    forces = stage2.forces
    friction = stage2.friction
    losses = stage2.losses
    flow = stage2.flow

    # Динамика (опционально)
    Kxx, Kyy, Cxx, Cyy = None, None, None, None
    margin, whirl_ratio, is_stable = None, None, None

    if compute_dynamics:
        coeffs = compute_dynamic_coefficients(
            base_config, eq.epsilon, eq.phi0,
            delta_e=0.01, delta_v_star=0.01,
            n_phi=180, n_z=50, verbose=False,
            film_model_factory=film_model_factory
        )
        Kxx, Kyy = coeffs.Kxx, coeffs.Kyy
        Cxx, Cyy = coeffs.Cxx, coeffs.Cyy

        stab = analyze_stability(coeffs.K, coeffs.C, BASE_MASS, BASE_N_RPM)
        margin = stab.stability_margin
        whirl_ratio = stab.whirl_ratio
        is_stable = stab.is_stable

    return CaseResult(
        name=name,
        epsilon=eq.epsilon,
        phi0_deg=np.degrees(eq.phi0),
        W=forces.W,
        p_max=reynolds.p_max,
        h_min=reynolds.h_min,
        Q=flow.Q_total,
        f=friction.mu_friction,
        P_loss=losses.P_friction,
        Kxx=Kxx, Kyy=Kyy, Cxx=Cxx, Cyy=Cyy,
        margin=margin, whirl_ratio=whirl_ratio, is_stable=is_stable,
        frac_lambda_lt_1=frac_lt_1,
        is_valid=is_valid,
        c=base_config.c,
        mu=base_config.mu,
        Fn=tex_params.Fn if tex_params else 0,
        h_star=tex_params.h_star if tex_params and tex_params.h_star else 0,
        pattern=tex_params.pattern if tex_params else '',
    )


# ============================================================================
# БЛОК 1: Сравнение A/B/C/D (база)
# ============================================================================

def run_block1_four_modes():
    """Сравнение 4-х режимов на базовой точке."""
    print("\n" + "=" * 70)
    print("БЛОК 1: Сравнение 4-х режимов (базовая точка)")
    print("=" * 70)
    print(f"c = {BASE_C*1e6:.0f} мкм, n = {BASE_N_RPM} об/мин, W_ext = {BASE_W_EXT/1000:.0f} кН")
    print(f"T = {BASE_TEMP}°C (μ = {BASE_MU} Па·с)")

    base_config = create_config(c=BASE_C, mu=BASE_MU)

    modes = [
        ('A', 'Гладкий', smooth_factory, False),
        ('B', 'Гладкий+Ra', smooth_factory, True),
        ('C', 'Текстура', texture_factory_gost, False),
        ('D', 'Текстура+Ra', texture_factory_gost, True),
    ]

    results = {}
    for key, name, factory, use_rough in modes:
        print(f"\n--- Режим {key}: {name} ---")
        result = compute_case(
            name=f"{key}_{name}",
            base_config=base_config,
            film_model_factory=factory,
            use_roughness=use_rough,
            roughness_params=ROUGHNESS_PARAMS if use_rough else None,
            compute_dynamics=True,
            tex_params=GOST_TEXTURE if 'Текстура' in name else None,
        )
        results[key] = result

        print(f"  ε = {result.epsilon:.4f}, h_min = {result.h_min*1e6:.2f} мкм")
        print(f"  p_max = {result.p_max/1e6:.2f} МПа, f = {result.f:.5f}")
        print(f"  P_loss = {result.P_loss:.1f} Вт, Q = {result.Q*1e6:.2f} см³/с")
        if result.Kxx:
            print(f"  Kxx = {result.Kxx/1e6:.0f} МН/м, Cxx = {result.Cxx/1e3:.0f} кН·с/м")
            print(f"  margin = {result.margin:.1f} 1/с, γ = {result.whirl_ratio:.3f}")

    # Сохраняем
    df = pd.DataFrame([{
        'mode': k,
        'epsilon': r.epsilon,
        'h_min_um': r.h_min*1e6,
        'p_max_MPa': r.p_max/1e6,
        'f': r.f,
        'P_loss_W': r.P_loss,
        'Q_cm3s': r.Q*1e6,
        'Kxx_MNm': r.Kxx/1e6 if r.Kxx else None,
        'Cxx_kNsm': r.Cxx/1e3 if r.Cxx else None,
        'margin': r.margin,
        'whirl_ratio': r.whirl_ratio,
    } for k, r in results.items()])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'four_modes_comparison.csv', index=False)
    print(f"\nCSV сохранён: {OUT_DIR / 'four_modes_comparison.csv'}")

    return results


# ============================================================================
# БЛОК 2: Sweep по зазору c
# ============================================================================

def run_block2_clearance_sweep():
    """Sweep по зазору c для режимов A и C."""
    print("\n" + "=" * 70)
    print("БЛОК 2: Sweep по зазору c")
    print("=" * 70)

    results = []

    for c in C_VALUES:
        print(f"\nc = {c*1e6:.0f} мкм:")

        base_config = create_config(c=c, mu=BASE_MU)

        # Обновляем текстуру с правильным h_star
        tex_params = TextureParams(
            a=0.5e-3,
            b=0.5e-3,
            h_star=0.10,  # безразмерная!
            Fn=0.15,
            pattern='phyllotaxis',
        )

        for mode, name, factory, tex in [
            ('A', 'Гладкий', smooth_factory, None),
            ('C', 'Текстура', create_texture_factory(tex_params), tex_params),
        ]:
            result = compute_case(
                name=f"{mode}_{name}_c{c*1e6:.0f}",
                base_config=base_config,
                film_model_factory=factory,
                compute_dynamics=True,
                tex_params=tex,
            )
            result.c = c
            results.append(result)

            print(f"  {mode}: ε={result.epsilon:.4f}, h_min={result.h_min*1e6:.2f} мкм, "
                  f"f={result.f:.5f}, P_loss={result.P_loss:.1f} Вт")

    # Сохраняем
    df = pd.DataFrame([{
        'c_um': r.c*1e6,
        'mode': r.name.split('_')[0],
        'epsilon': r.epsilon,
        'h_min_um': r.h_min*1e6,
        'p_max_MPa': r.p_max/1e6,
        'f': r.f,
        'P_loss_W': r.P_loss,
        'Q_cm3s': r.Q*1e6,
        'margin': r.margin,
    } for r in results])

    df.to_csv(OUT_DIR / 'clearance_sweep.csv', index=False)
    print(f"\nCSV сохранён: {OUT_DIR / 'clearance_sweep.csv'}")

    # График
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    df_A = df[df['mode'] == 'A']
    df_C = df[df['mode'] == 'C']

    metrics = [
        ('epsilon', 'ε', axes[0, 0]),
        ('h_min_um', 'h_min, мкм', axes[0, 1]),
        ('p_max_MPa', 'p_max, МПа', axes[0, 2]),
        ('f', 'f (коэф. трения)', axes[1, 0]),
        ('P_loss_W', 'P_loss, Вт', axes[1, 1]),
        ('margin', 'margin, 1/с', axes[1, 2]),
    ]

    for col, label, ax in metrics:
        ax.plot(df_A['c_um'], df_A[col], 'o-', label='A (Гладкий)')
        ax.plot(df_C['c_um'], df_C[col], 's--', label='C (Текстура)')
        ax.set_xlabel('c, мкм')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'clearance_sweep.png', dpi=150)
    print(f"График сохранён: {OUT_DIR / 'clearance_sweep.png'}")
    plt.close()

    return results


# ============================================================================
# БЛОК 3: Оптимизация текстуры (ПАРАЛЛЕЛЬНО)
# ============================================================================

def _compute_texture_case_fast(args: Tuple) -> CaseResult:
    """
    БЫСТРЫЙ воркер: использует фиксированное ε вместо find_equilibrium.

    Идея: текстура слабо влияет на положение равновесия, поэтому
    используем ε от гладкого подшипника и сравниваем только p_max, P_loss.
    """
    Fn, h_star, pattern, phi_min, phi_max, base_epsilon, base_phi0 = args
    sector_name = 'full' if phi_max - phi_min > 6 else 'partial'

    tex_params = TextureParams(
        a=0.5e-3,
        b=0.5e-3,
        h_star=h_star,
        Fn=Fn,
        pattern=pattern,
        phi_min=phi_min,
        phi_max=phi_max,
    )

    # Конфиг с фиксированным ε (от гладкого подшипника)
    config = BearingConfig(
        R=BASE_R, L=BASE_L, c=BASE_C,
        epsilon=base_epsilon, phi0=base_phi0,
        n_rpm=BASE_N_RPM, mu=BASE_MU,
        n_phi=180, n_z=50
    )

    # Проверка валидности текстуры
    film_model = TexturedFilmModel(config, tex_params)
    if not film_model.is_valid:
        return CaseResult(
            name=f"Fn{Fn}_h{h_star}_{pattern}_{sector_name}",
            epsilon=base_epsilon, phi0_deg=np.degrees(base_phi0),
            W=0, p_max=0, h_min=0, Q=0, f=0, P_loss=0,
            is_valid=False, Fn=Fn, h_star=h_star, pattern=pattern,
        )

    # Один вызов solver (без итераций!)
    reynolds = solve_reynolds(config, film_model)
    stage2 = compute_stage2(reynolds, config)

    return CaseResult(
        name=f"Fn{Fn}_h{h_star}_{pattern}_{sector_name}",
        epsilon=base_epsilon,
        phi0_deg=np.degrees(base_phi0),
        W=stage2.forces.W,
        p_max=reynolds.p_max,
        h_min=reynolds.h_min,
        Q=stage2.flow.Q_total,
        f=stage2.friction.mu_friction,
        P_loss=stage2.losses.P_friction,
        is_valid=True,
        Fn=Fn, h_star=h_star, pattern=pattern,
    )


def _compute_texture_case(args: Tuple) -> CaseResult:
    """
    Воркер для параллельного расчёта одной конфигурации текстуры.

    Функция на уровне модуля для совместимости с ProcessPoolExecutor (pickle).
    """
    Fn, h_star, pattern, phi_min, phi_max = args
    sector_name = 'full' if phi_max - phi_min > 6 else 'partial'

    base_config = create_config(c=BASE_C, mu=BASE_MU)

    tex_params = TextureParams(
        a=0.5e-3,
        b=0.5e-3,
        h_star=h_star,
        Fn=Fn,
        pattern=pattern,
        phi_min=phi_min,
        phi_max=phi_max,
    )

    # Предварительная проверка валидности
    test_model = TexturedFilmModel(base_config, tex_params)
    if not test_model.is_valid:
        return CaseResult(
            name=f"Fn{Fn}_h{h_star}_{pattern}_{sector_name}",
            epsilon=0, phi0_deg=0, W=0, p_max=0, h_min=0,
            Q=0, f=0, P_loss=0, is_valid=False,
            Fn=Fn, h_star=h_star, pattern=pattern,
        )

    factory = create_texture_factory(tex_params)

    result = compute_case(
        name=f"Fn{Fn}_h{h_star}_{pattern}_{sector_name}",
        base_config=base_config,
        film_model_factory=factory,
        compute_dynamics=False,  # Для скорости
        tex_params=tex_params,
    )
    return result


def run_block3_texture_optimization(parallel: bool = True, max_workers: int = None, fast_mode: bool = True):
    """
    Сетка оптимизации текстуры.

    Args:
        parallel: использовать параллельные вычисления (по умолчанию True)
        max_workers: количество процессов (по умолчанию cpu_count - 1)
        fast_mode: использовать фиксированное ε вместо find_equilibrium (в 30 раз быстрее)
    """
    print("\n" + "=" * 70)
    print("БЛОК 3: Оптимизация текстуры")
    print("=" * 70)

    # В fast_mode сначала находим ε для гладкого подшипника
    base_epsilon = 0.6
    base_phi0 = np.radians(135)

    if fast_mode:
        print("  Быстрый режим: поиск базового ε...")
        base_config = create_config(c=BASE_C, mu=BASE_MU)
        eq = find_equilibrium(
            base_config, W_ext=BASE_W_EXT, load_angle=-np.pi/2,
            verbose=False, film_model_factory=smooth_factory
        )
        base_epsilon = eq.epsilon
        base_phi0 = eq.phi0
        print(f"  Базовое ε = {base_epsilon:.4f}, φ₀ = {np.degrees(base_phi0):.1f}°")

    # Формируем список задач
    tasks = []
    for Fn in FN_VALUES:
        for h_star in H_STAR_VALUES:
            for pattern in PATTERN_VALUES:
                for phi_min, phi_max in SECTOR_VALUES:
                    if fast_mode:
                        tasks.append((Fn, h_star, pattern, phi_min, phi_max, base_epsilon, base_phi0))
                    else:
                        tasks.append((Fn, h_star, pattern, phi_min, phi_max))

    total = len(tasks)
    print(f"Всего конфигураций: {total}")

    # Выбираем воркер
    worker_fn = _compute_texture_case_fast if fast_mode else _compute_texture_case

    # Прогрев Numba JIT
    print("  Прогрев Numba JIT...")
    _ = worker_fn(tasks[0])
    print("  Прогрев завершён")

    if parallel:
        # Параллельное выполнение
        n_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        print(f"Параллельный режим: {n_workers} процессов")

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Запускаем все задачи
            future_to_task = {executor.submit(worker_fn, task): task for task in tasks}

            # Собираем результаты по мере готовности
            for i, future in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[future]
                try:
                    # Таймаут 60 секунд на одну конфигурацию
                    result = future.result(timeout=60)
                    results.append(result)
                    if i % 10 == 0 or i == total:
                        print(f"  [{i}/{total}] завершено")
                except TimeoutError:
                    Fn, h_star, pattern, phi_min, phi_max = task
                    sector_name = 'full' if phi_max - phi_min > 6 else 'partial'
                    print(f"  [TIMEOUT] Fn={Fn}, h*={h_star}, {pattern}, {sector_name}")
                    results.append(CaseResult(
                        name=f"Fn{Fn}_h{h_star}_{pattern}_{sector_name}",
                        epsilon=0, phi0_deg=0, W=0, p_max=0, h_min=0,
                        Q=0, f=0, P_loss=0, is_valid=False,
                        Fn=Fn, h_star=h_star, pattern=pattern,
                    ))
                except Exception as e:
                    Fn, h_star, pattern, phi_min, phi_max = task
                    sector_name = 'full' if phi_max - phi_min > 6 else 'partial'
                    print(f"  ОШИБКА: Fn={Fn}, h*={h_star}, {pattern}, {sector_name}: {e}")
                    results.append(CaseResult(
                        name=f"Fn{Fn}_h{h_star}_{pattern}_{sector_name}",
                        epsilon=0, phi0_deg=0, W=0, p_max=0, h_min=0,
                        Q=0, f=0, P_loss=0, is_valid=False,
                        Fn=Fn, h_star=h_star, pattern=pattern,
                    ))
    else:
        # Последовательное выполнение (для отладки)
        print("Последовательный режим")
        results = []
        for i, task in enumerate(tasks, 1):
            result = worker_fn(task)
            results.append(result)
            if i % 10 == 0:
                Fn, h_star, pattern, _, _ = task
                print(f"  [{i}/{total}] Fn={Fn:.0%}, h*={h_star}, {pattern}: "
                      f"valid={result.is_valid}, P_loss={result.P_loss:.1f} Вт")

    # Сохраняем
    df = pd.DataFrame([{
        'Fn': r.Fn,
        'h_star': r.h_star,
        'pattern': r.pattern,
        'is_valid': r.is_valid,
        'epsilon': r.epsilon,
        'h_min_um': r.h_min*1e6,
        'p_max_MPa': r.p_max/1e6,
        'f': r.f,
        'P_loss_W': r.P_loss,
        'Q_cm3s': r.Q*1e6 if r.Q else 0,
    } for r in results])

    df.to_csv(OUT_DIR / 'texture_optimization.csv', index=False)
    print(f"\nCSV сохранён: {OUT_DIR / 'texture_optimization.csv'}")

    return results


# ============================================================================
# БЛОК 4: Pareto-анализ
# ============================================================================

def run_block4_pareto(optimization_results: List[CaseResult]):
    """Pareto-анализ: p_max vs P_loss."""
    print("\n" + "=" * 70)
    print("БЛОК 4: Pareto-анализ")
    print("=" * 70)

    # Базовый гладкий для сравнения
    base_config = create_config()
    base_result = compute_case(
        name='baseline_smooth',
        base_config=base_config,
        film_model_factory=smooth_factory,
    )

    # Фильтруем валидные с ограничениями
    valid = [r for r in optimization_results
             if r.is_valid and r.h_min >= 10e-6 and r.W > 0.9 * BASE_W_EXT]

    print(f"Валидных конфигураций: {len(valid)} из {len(optimization_results)}")

    if len(valid) == 0:
        print("Нет валидных конфигураций для Pareto-анализа!")
        return []

    # Pareto-фронт: минимизируем p_max и P_loss
    def is_pareto_optimal(candidate, others):
        for other in others:
            if other.p_max <= candidate.p_max and other.P_loss <= candidate.P_loss:
                if other.p_max < candidate.p_max or other.P_loss < candidate.P_loss:
                    return False
        return True

    pareto = [r for r in valid if is_pareto_optimal(r, valid)]
    print(f"Точек на Pareto-фронте: {len(pareto)}")

    # Топ-5 по P_loss (с учётом h_min)
    top5 = sorted(valid, key=lambda r: r.P_loss)[:5]

    print("\nТоп-5 по P_loss:")
    for i, r in enumerate(top5, 1):
        print(f"  {i}. {r.name}: P_loss={r.P_loss:.1f} Вт, "
              f"p_max={r.p_max/1e6:.2f} МПа, h_min={r.h_min*1e6:.2f} мкм")

    # Сохраняем
    df = pd.DataFrame([{
        'rank': i+1,
        'name': r.name,
        'Fn': r.Fn,
        'h_star': r.h_star,
        'pattern': r.pattern,
        'epsilon': r.epsilon,
        'h_min_um': r.h_min*1e6,
        'p_max_MPa': r.p_max/1e6,
        'f': r.f,
        'P_loss_W': r.P_loss,
    } for i, r in enumerate(top5)])

    df.to_csv(OUT_DIR / 'pareto_top5.csv', index=False)
    print(f"\nCSV сохранён: {OUT_DIR / 'pareto_top5.csv'}")

    # График Pareto
    fig, ax = plt.subplots(figsize=(10, 8))

    # Все валидные точки
    p_max_all = [r.p_max/1e6 for r in valid]
    P_loss_all = [r.P_loss for r in valid]
    ax.scatter(P_loss_all, p_max_all, c='lightgray', s=30, alpha=0.5, label='Все валидные')

    # Pareto-фронт
    if pareto:
        p_max_pareto = [r.p_max/1e6 for r in pareto]
        P_loss_pareto = [r.P_loss for r in pareto]
        ax.scatter(P_loss_pareto, p_max_pareto, c='blue', s=80, marker='s', label='Pareto-фронт')

    # Топ-5
    p_max_top = [r.p_max/1e6 for r in top5]
    P_loss_top = [r.P_loss for r in top5]
    ax.scatter(P_loss_top, p_max_top, c='red', s=120, marker='*', label='Топ-5')

    # Базовый гладкий
    ax.axhline(base_result.p_max/1e6, color='green', linestyle='--', label=f'Гладкий p_max')
    ax.axvline(base_result.P_loss, color='green', linestyle=':', label=f'Гладкий P_loss')

    ax.set_xlabel('P_loss, Вт')
    ax.set_ylabel('p_max, МПа')
    ax.set_title('Pareto-анализ текстуры')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pareto_analysis.png', dpi=150)
    print(f"График сохранён: {OUT_DIR / 'pareto_analysis.png'}")
    plt.close()

    return top5


# ============================================================================
# БЛОК 5: Робастность топ-5
# ============================================================================

def run_block5_robustness(top5: List[CaseResult], skip: bool = True):
    """Робастность топ-5 по c и μ."""
    print("\n" + "=" * 70)
    print("БЛОК 5: Робастность топ-5")
    print("=" * 70)

    if skip:
        print("  [ПРОПУЩЕН] — используйте run_block5_robustness(top5, skip=False) для полного анализа")
        return []

    if not top5:
        print("Нет топ-5 для анализа робастности!")
        return

    results = []

    # Sweep по c для топ-5
    c_subset = [30e-6, 50e-6, 100e-6, 180e-6]

    for r in top5[:3]:  # Только первые 3 для скорости
        print(f"\n{r.name}:")

        tex_params = TextureParams(
            a=0.5e-3,
            b=0.5e-3,
            h_star=r.h_star,
            Fn=r.Fn,
            pattern=r.pattern,
        )

        for c in c_subset:
            base_config = create_config(c=c, mu=BASE_MU)
            factory = create_texture_factory(tex_params)

            result = compute_case(
                name=f"{r.name}_c{c*1e6:.0f}",
                base_config=base_config,
                film_model_factory=factory,
                compute_dynamics=True,
                tex_params=tex_params,
            )
            result.c = c
            results.append(result)

            print(f"  c={c*1e6:.0f} мкм: ε={result.epsilon:.4f}, "
                  f"P_loss={result.P_loss:.1f} Вт, margin={result.margin:.1f} 1/с")

    # Сохраняем
    if results:
        df = pd.DataFrame([{
            'name': r.name,
            'c_um': r.c*1e6,
            'epsilon': r.epsilon,
            'h_min_um': r.h_min*1e6,
            'p_max_MPa': r.p_max/1e6,
            'P_loss_W': r.P_loss,
            'margin': r.margin,
        } for r in results])

        df.to_csv(OUT_DIR / 'robustness_sweep.csv', index=False)
        print(f"\nCSV сохранён: {OUT_DIR / 'robustness_sweep.csv'}")

    return results


# ============================================================================
# БЛОК 6: Итоговые графики
# ============================================================================

def run_block6_summary():
    """Итоговые графики и таблицы."""
    print("\n" + "=" * 70)
    print("БЛОК 6: Итоговые графики")
    print("=" * 70)

    # Heatmap: P_loss vs (Fn, h_star)
    try:
        df = pd.read_csv(OUT_DIR / 'texture_optimization.csv')
        df_valid = df[df['is_valid'] == True]

        if len(df_valid) > 0:
            # Pivot для heatmap (усредняем по pattern и sector)
            pivot = df_valid.groupby(['Fn', 'h_star'])['P_loss_W'].mean().reset_index()
            pivot_table = pivot.pivot(index='h_star', columns='Fn', values='P_loss_W')

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pivot_table.values, cmap='viridis_r', aspect='auto')
            ax.set_xticks(range(len(pivot_table.columns)))
            ax.set_xticklabels([f'{x:.0%}' for x in pivot_table.columns])
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_yticklabels([f'{x:.2f}' for x in pivot_table.index])
            ax.set_xlabel('Fn')
            ax.set_ylabel('h*')
            ax.set_title('P_loss (Вт) vs (Fn, h*)')
            plt.colorbar(im, ax=ax, label='P_loss, Вт')

            plt.tight_layout()
            plt.savefig(OUT_DIR / 'heatmap_P_loss.png', dpi=150)
            print(f"Heatmap сохранён: {OUT_DIR / 'heatmap_P_loss.png'}")
            plt.close()

    except Exception as e:
        print(f"Ошибка при создании heatmap: {e}")

    print("\nИтоговые файлы в", OUT_DIR)


# ============================================================================
# MAIN
# ============================================================================

def run_diagnostic_test():
    """
    Диагностический тест: один расчёт как в Stage 7.
    Если тут работает, а в БЛОК 1 нет — проблема в обвязке.
    """
    print("\n" + "=" * 70)
    print("ДИАГНОСТИКА: Тест одной точки (как Stage 7)")
    print("=" * 70)

    from bearing_solver import compute_stage2

    # Параметры как в Stage 7
    config = BearingConfig(
        R=0.050,
        L=0.050,
        c=50e-6,
        epsilon=0.6,
        phi0=np.radians(45),
        n_rpm=3000,
        mu=0.04,  # как в Stage 7
        n_phi=180,
        n_z=50,
    )

    print(f"Конфиг: R={config.R}, L={config.L}, c={config.c*1e6:.0f} мкм")
    print(f"        ε={config.epsilon}, φ₀={np.degrees(config.phi0):.1f}°")
    print(f"        n={config.n_rpm} об/мин, μ={config.mu} Па·с")

    # 1) Гладкая модель без равновесия
    print("\n--- Тест 1: Гладкая модель (без равновесия) ---")
    film_model = SmoothFilmModel(config)
    reynolds = solve_reynolds(config, film_model)
    forces = compute_forces(reynolds, config)

    print(f"  p: max={reynolds.p_max/1e6:.2f} МПа, h_min={reynolds.h_min*1e6:.2f} мкм")
    print(f"  W = {forces.W/1000:.2f} кН")

    if reynolds.p_max < 1e3:
        print("  [ОШИБКА] p_max слишком мал! Solver не работает.")
        return False

    # 2) Равновесие для гладкой модели
    print("\n--- Тест 2: Гладкая модель + find_equilibrium ---")
    W_ext = 50e3  # 50 кН

    try:
        eq = find_equilibrium(
            config, W_ext=W_ext, load_angle=-np.pi/2,
            verbose=True, film_model_factory=smooth_factory
        )
        print(f"  Равновесие найдено: ε={eq.epsilon:.4f}, φ₀={np.degrees(eq.phi0):.1f}°")
        print(f"  W_achieved = {eq.W_achieved/1000:.2f} кН (цель: {W_ext/1000:.0f} кН)")
        print(f"  residual_vec = {eq.residual_vec*100:.2f}%")
    except Exception as e:
        print(f"  [ОШИБКА] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3) С параметрами Stage 8
    print("\n--- Тест 3: Параметры Stage 8 (μ=0.057) ---")
    config_s8 = create_config(c=BASE_C, mu=BASE_MU)
    print(f"  c={config_s8.c*1e6:.0f} мкм, μ={config_s8.mu} Па·с")

    try:
        eq_s8 = find_equilibrium(
            config_s8, W_ext=BASE_W_EXT, load_angle=-np.pi/2,
            verbose=True, film_model_factory=smooth_factory
        )
        print(f"  Равновесие Stage8: ε={eq_s8.epsilon:.4f}, W={eq_s8.W_achieved/1000:.2f} кН")
    except Exception as e:
        print(f"  [ОШИБКА] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[OK] Диагностика пройдена — solver работает!")
    return True


def main(run_diagnostics: bool = False):
    """
    Запуск этапа 8.

    Args:
        run_diagnostics: запустить диагностический тест перед основным расчётом
    """
    print("ЭТАП 8: Параметрические исследования и оптимизация текстуры")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Диагностика (опционально)
    if run_diagnostics:
        if not run_diagnostic_test():
            print("\n[КРИТИЧЕСКАЯ ОШИБКА] Диагностика не пройдена!")
            return

    # БЛОК 1
    four_modes = run_block1_four_modes()

    # БЛОК 2
    clearance_results = run_block2_clearance_sweep()

    # БЛОК 3
    optimization_results = run_block3_texture_optimization()

    # БЛОК 4
    top5 = run_block4_pareto(optimization_results)

    # БЛОК 5
    robustness_results = run_block5_robustness(top5)

    # БЛОК 6
    run_block6_summary()

    print("\n" + "=" * 70)
    print("ЭТАП 8: ЗАВЕРШЁН")
    print(f"Результаты в: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    run_diag = '--diag' in sys.argv or '--diagnostics' in sys.argv
    main(run_diagnostics=run_diag)
