"""
Construção do vetor de regressores phi[k] do ARX e do estimador de
volatilidade. Como a Itaúsa é uma holding financeira dominada pelo Itaú, o
conjunto de variáveis exógenas é adaptado ao seu perfil: ITUB4 (fator
dominante), EWZ (fluxo estrangeiro), nível do CDI (juros), além de IBOV,
USD/BRL e VIX; completam o vetor os retornos defasados da própria ITSA4 (a
parte autorregressiva, de onde saem os polos de G(z)) e indicadores de
volatilidade, tendência e volume. O alvo é o retorno log da semana seguinte,
r[k+1]. A previsão de volatilidade é causal: a EWMA da variância
(RiskMetrics, sigma^2[k] = lambda_v sigma^2[k-1] + (1 - lambda_v) r^2[k], um
filtro passa-baixas de primeira ordem com polo lambda_v) usa apenas retornos
até k, com ajuste opcional pelo nível relativo do VIX.
"""

import numpy as np
import pandas as pd


def obter_lista_features() -> list:
    return [
        "r_itsa4_k", "r_itsa4_k_1", "r_itsa4_k_2", "r_itsa4_k_3",
        "vol_itsa4_4s", "vol_itsa4_12s", "volume_relativo_itsa4", "tendencia_itsa4_12s",
        "r_ibov_k", "r_ibov_k_1", "r_ibov_k_2", "vol_ibov_4s", "tendencia_ibov_12s",
        "r_usdbrl_k", "r_usdbrl_k_1",
        "r_itub4_k", "r_itub4_k_1", "r_itub4_k_2",
        "r_ewz_k", "r_ewz_k_1",
        "vix_nivel_k", "r_vix_k", "r_vix_k_1",
        "nivel_cdi_k",
    ]


def _tendencia(preco: pd.Series, janela: int = 12) -> pd.Series:
    return preco / preco.rolling(janela).mean() - 1


def criar_features_arx(retornos, dados_semanais, cdi_semanal) -> pd.DataFrame:
    idx = retornos.index
    df = pd.DataFrame(index=idx)

    r = retornos["ITSA4"]
    for atraso in range(4):
        df[f"r_itsa4_k{'' if atraso == 0 else f'_{atraso}'}"] = r.shift(atraso)
    df["vol_itsa4_4s"] = r.rolling(4).std()
    df["vol_itsa4_12s"] = r.rolling(12).std()
    volume = dados_semanais["Volume_ITSA4"].reindex(idx)
    df["volume_relativo_itsa4"] = volume / volume.rolling(12).mean()
    df["tendencia_itsa4_12s"] = _tendencia(dados_semanais["ITSA4"].reindex(idx))

    df["r_ibov_k"] = retornos["IBOV"]
    df["r_ibov_k_1"] = retornos["IBOV"].shift(1)
    df["r_ibov_k_2"] = retornos["IBOV"].shift(2)
    df["vol_ibov_4s"] = retornos["IBOV"].rolling(4).std()
    df["tendencia_ibov_12s"] = _tendencia(dados_semanais["IBOV"].reindex(idx))

    df["r_usdbrl_k"] = retornos["USDBRL"]
    df["r_usdbrl_k_1"] = retornos["USDBRL"].shift(1)

    df["r_itub4_k"] = retornos["ITUB4"]
    df["r_itub4_k_1"] = retornos["ITUB4"].shift(1)
    df["r_itub4_k_2"] = retornos["ITUB4"].shift(2)

    df["r_ewz_k"] = retornos["EWZ"]
    df["r_ewz_k_1"] = retornos["EWZ"].shift(1)

    df["vix_nivel_k"] = dados_semanais["VIX"].reindex(idx)
    df["r_vix_k"] = retornos["VIX"]
    df["r_vix_k_1"] = retornos["VIX"].shift(1)

    df["nivel_cdi_k"] = cdi_semanal.reindex(idx)

    df["target"] = r.shift(-1)
    return df.dropna()


def prever_volatilidade(retornos_itsa4, vix_nivel=None, lambda_ewma=0.94,
                        usar_vix=True, peso_vix=0.5, periodos_por_ano=52) -> pd.Series:
    r = pd.Series(retornos_itsa4).astype(float)
    r2 = r.values ** 2
    variancia = np.empty(len(r2))
    variancia[0] = r2[0]
    for k in range(1, len(r2)):
        variancia[k] = lambda_ewma * variancia[k - 1] + (1 - lambda_ewma) * r2[k]
    sigma = pd.Series(np.sqrt(variancia) * np.sqrt(periodos_por_ano), index=r.index)

    if usar_vix and vix_nivel is not None:
        vix = pd.Series(vix_nivel, index=r.index).astype(float)
        fator = 1.0 + peso_vix * (vix / vix.rolling(52, min_periods=4).mean() - 1.0)
        sigma = sigma * fator.clip(0.5, 2.0).fillna(1.0)
    return sigma
