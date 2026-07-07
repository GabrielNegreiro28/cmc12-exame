"""
Métricas de desempenho e significância estatística. O índice de Sharpe é
sempre calculado sobre o excesso de retorno em relação ao CDI e anualizado
por sqrt(P), com P = 52 semanas. A significância usa a estatística t do
excesso semanal e bootstrap de blocos móveis (que preserva a autocorrelação
dos retornos) para construir o intervalo de confiança do Sharpe e os
p-valores de Sharpe <= 0 e de não superar o benchmark. A qualidade do
previsor ARX é medida por RMSE, MAE, acurácia direcional e correlação com o
retorno realizado.
"""

import numpy as np
import pandas as pd
from scipy import stats


def calcular_drawdown(capital) -> pd.Series:
    capital = pd.Series(capital)
    return capital / capital.cummax() - 1


def _sharpe_excesso(retornos, caixa, periodos_por_ano):
    excesso = np.asarray(retornos) - np.asarray(caixa)
    desvio = excesso.std(ddof=1) if len(excesso) > 1 else 0.0
    return np.nan if desvio == 0 else (excesso.mean() / desvio) * np.sqrt(periodos_por_ano)


def calcular_metricas(resultado, periodos_por_ano=52) -> dict:
    capital = resultado["capital"]
    retornos = resultado["retorno_carteira"]
    caixa = resultado.get("retorno_caixa", pd.Series(0.0, index=resultado.index))

    retorno_total = capital.iloc[-1] - 1
    return {
        "capital_final": capital.iloc[-1],
        "retorno_total": retorno_total,
        "retorno_anualizado": (1 + retorno_total) ** (periodos_por_ano / len(retornos)) - 1,
        "retorno_cdi_total": np.prod(1 + caixa.values) - 1,
        "volatilidade_anualizada": retornos.std() * np.sqrt(periodos_por_ano),
        "sharpe": _sharpe_excesso(retornos.values, caixa.values, periodos_por_ano),
        "max_drawdown": calcular_drawdown(capital).min(),
        "posicao_media": resultado["posicao_u"].mean(),
        "troca_media": resultado["posicao_u"].diff().abs().mean(),
    }


def avaliar_previsor(retorno_real, retorno_previsto) -> dict:
    real = np.asarray(retorno_real, dtype=float)
    previsto = np.asarray(retorno_previsto, dtype=float)
    erro = real - previsto
    correlacao = (np.nan if np.std(previsto) == 0
                  else float(np.corrcoef(real, previsto)[0, 1]))
    return {
        "rmse": float(np.sqrt(np.mean(erro ** 2))),
        "mae": float(np.mean(np.abs(erro))),
        "acuracia_direcional": float(np.mean(np.sign(real) == np.sign(previsto))),
        "correlacao": correlacao,
    }


def _bloco_bootstrap(n, tamanho, rng):
    indices = []
    while len(indices) < n:
        inicio = rng.integers(0, max(1, n - tamanho + 1))
        indices.extend(range(inicio, min(inicio + tamanho, n)))
    return np.array(indices[:n])


def _percentil(amostras, q):
    return np.nan if len(amostras) == 0 else np.percentile(amostras, q)


def calcular_significancia(resultado, benchmark=None, periodos_por_ano=52,
                           n_bootstrap=5000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    retornos = resultado["retorno_carteira"].values
    caixa = resultado["retorno_caixa"].values
    excesso = retornos - caixa
    n = len(excesso)
    tamanho_bloco = max(1, round(n ** (1 / 3)))

    desvio = excesso.std(ddof=1) if n > 1 else 0.0
    t_stat = np.nan if desvio == 0 else excesso.mean() / (desvio / np.sqrt(n))
    p_t = np.nan if np.isnan(t_stat) else float(stats.t.sf(abs(t_stat), n - 1) * 2)
    sharpe = _sharpe_excesso(retornos, caixa, periodos_por_ano)

    tem_benchmark = benchmark is not None
    if tem_benchmark:
        ret_b = benchmark["retorno_carteira"].values
        caixa_b = benchmark["retorno_caixa"].values
        sharpe_b = _sharpe_excesso(ret_b, caixa_b, periodos_por_ano)

    sharpes_boot, diferencas_boot = [], []
    for _ in range(n_bootstrap):
        idx = _bloco_bootstrap(n, tamanho_bloco, rng)
        s = _sharpe_excesso(retornos[idx], caixa[idx], periodos_por_ano)
        if not np.isnan(s):
            sharpes_boot.append(s)
            if tem_benchmark:
                diferencas_boot.append(
                    s - _sharpe_excesso(ret_b[idx], caixa_b[idx], periodos_por_ano))
    sharpes_boot = np.array(sharpes_boot)
    diferencas_boot = np.array(diferencas_boot)

    saida = {
        "n_observacoes": n, "t_stat_excesso": t_stat, "p_valor_t": p_t,
        "sharpe_observado": sharpe,
        "sharpe_ic95_inf": _percentil(sharpes_boot, 2.5),
        "sharpe_ic95_sup": _percentil(sharpes_boot, 97.5),
        "p_valor_sharpe_menor_igual_zero":
            np.mean(sharpes_boot <= 0) if len(sharpes_boot) else np.nan,
    }
    if tem_benchmark:
        saida.update({
            "sharpe_benchmark": sharpe_b, "diff_sharpe": sharpe - sharpe_b,
            "p_valor_nao_superar_benchmark":
                np.mean(diferencas_boot <= 0) if len(diferencas_boot) else np.nan,
        })
    return saida
