"""
Simulação da malha de controle e procedimentos de avaliação. A planta é a
dinâmica de capital C[k+1] = C[k] (1 + u[k] r[k+1] + (1 - u[k]) r_cdi[k+1]
- c |u[k] - u[k-1]|), aplicada por um único simulador a qualquer estratégia
que respeite o protocolo Estrategia. A seleção de hiperparâmetros segue a
metodologia do tutorial de otimização de CMC-12: uma busca em grade fornece o
chute inicial e o método de Nelder-Mead (equivalente ao fminsearch do MATLAB,
via scipy) refina a solução, maximizando o índice de Sharpe na validação com
os parâmetros parametrizados em log10 e restritos à faixa coberta pela grade
(fora dela o custo é infinito), o que evita soluções degeneradas como rho -> 0. O experimento de atraso desloca a
previsão e as features em n semanas para medir a degradação causada por
atraso de transporte na malha, e o walk-forward re-identifica o modelo em
janela expansível usando apenas o passado: para prever o alvo da posição i
(retorno em i+1), o modelo é treinado somente com linhas até i-1, cujos
rótulos são conhecidos no instante i.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from estrategias import Passo
from metricas import calcular_metricas


def simular(estrategia, retornos_reais, retornos_previstos, vols_previstas,
            features, datas, custo, retornos_caixa) -> pd.DataFrame:
    n = len(retornos_reais)
    capital, posicoes, retornos_carteira = [1.0], [], []
    u_anterior = 0.0
    for k in range(n):
        u = estrategia.alocar(Passo(
            retorno_previsto=float(retornos_previstos[k]),
            vol_prevista=float(vols_previstas[k]),
            features=features.iloc[k],
            u_anterior=u_anterior,
        ))
        r = (u * retornos_reais[k] + (1 - u) * retornos_caixa[k]
             - custo * abs(u - u_anterior))
        capital.append(capital[-1] * (1 + r))
        posicoes.append(u)
        retornos_carteira.append(r)
        u_anterior = u

    return pd.DataFrame({
        "retorno_real_ativo": retornos_reais,
        "retorno_previsto": retornos_previstos,
        "retorno_caixa": retornos_caixa,
        "posicao_u": posicoes,
        "retorno_carteira": retornos_carteira,
        "capital": capital[1:],
    }, index=datas)


def varrer(fabrica_estrategia, grade, contexto, periodos_por_ano=52) -> pd.DataFrame:
    linhas = []
    for params in grade:
        resultado = simular(fabrica_estrategia(**params), **contexto)
        metricas = calcular_metricas(resultado, periodos_por_ano)
        linhas.append({**params, **metricas})
    return pd.DataFrame(linhas)


def refinar(fabrica_estrategia, params_iniciais, contexto, limites,
            periodos_por_ano=52):
    nomes = list(params_iniciais)

    def funcao_custo(x):
        params = dict(zip(nomes, np.power(10.0, x)))
        if any(not (limites[n][0] <= params[n] <= limites[n][1]) for n in nomes):
            return np.inf
        resultado = simular(fabrica_estrategia(**params), **contexto)
        sharpe = calcular_metricas(resultado, periodos_por_ano)["sharpe"]
        return np.inf if np.isnan(sharpe) else -sharpe

    x0 = np.log10([params_iniciais[nome] for nome in nomes])
    solucao = minimize(funcao_custo, x0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-4})
    params = {nome: float(10.0 ** v) for nome, v in zip(nomes, solucao.x)}
    return params, float(-solucao.fun)


def atrasar_previsoes(contexto, atraso: int) -> dict:
    if atraso == 0:
        return contexto
    previstos = np.asarray(contexto["retornos_previstos"])
    previstos = np.concatenate((np.full(atraso, previstos[0]), previstos[:-atraso]))
    features = contexto["features"].shift(atraso).bfill()
    return dict(contexto, retornos_previstos=previstos, features=features)


def walk_forward_previsoes(df, features, target, sigma_full, cdi_semanal, treinar,
                           inicio, passo_retreino=4, minimo_treino=80,
                           devolver_modelos=False):
    df = df.sort_index()
    X, y = df[features], df[target]
    n = len(df)

    mascara = np.asarray(df.index >= pd.to_datetime(inicio))
    i0 = max(int(np.argmax(mascara)) if mascara.any() else n, minimo_treino)
    if i0 >= n:
        raise ValueError("Sem observações suficientes para o walk-forward.")

    datas, reais, previstos, modelos = [], [], [], []
    modelo, desde_retreino = None, passo_retreino
    for i in range(i0, n):
        if modelo is None or desde_retreino >= passo_retreino:
            modelo = treinar(X.iloc[:i], y.iloc[:i])
            desde_retreino = 0
        previstos.append(float(modelo.predict(X.iloc[[i]])[0]))
        reais.append(float(y.iloc[i]))
        datas.append(df.index[i])
        modelos.append(modelo)
        desde_retreino += 1

    idx = pd.DatetimeIndex(datas)
    previsoes = pd.DataFrame({
        "retorno_real": np.expm1(reais),
        "retorno_previsto": np.expm1(previstos),
        "vol_prevista": sigma_full.reindex(idx).ffill().bfill().values,
        "cdi": cdi_semanal.reindex(idx).ffill().fillna(0.0).values,
    }, index=idx)
    return (previsoes, modelos) if devolver_modelos else previsoes
