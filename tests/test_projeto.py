"""
Suíte de verificação offline, executada sem internet sobre dados sintéticos.
Confere as peças centrais do projeto: o cálculo do Sharpe de excesso e da
significância; o simulador único (Buy and Hold reproduz o produto acumulado);
o resolvedor do QP do MPC contra busca exaustiva; a equivalência entre o MPC
de horizonte 1, o baseline analítico e a solução fechada; a previsão
multi-passo com o pipeline real; a ausência de vazamento temporal no
walk-forward (via treinador espião injetado); a extração de A(z), polos,
estabilidade e ganho DC da função de transferência; a causalidade do
estimador de volatilidade; a equivalência do controlador H=1 com seu filtro
de primeira ordem e a fórmula da banda passante de -3 dB; o uso do modelo
vigente pelo MPC no walk-forward; o refino por Nelder-Mead (nunca pior que o
chute inicial da grade); e a degradação de desempenho causada por atraso na
previsão. Rodar a partir da raiz do projeto: py tests\\test_projeto.py
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "src"))

from features import obter_lista_features, prever_volatilidade
from identificacao import treinar_modelo
from metricas import calcular_metricas, calcular_significancia
from estrategias import (ControleH1, ControleMPC, ControleMPCWF, VolTarget,
                         BuyAndHold, Passo, resolver_mpc, prever_horizonte)
import backtest as BT
import funcao_transferencia as TF

FALHAS = []


def checar(condicao, mensagem):
    print(("OK   " if condicao else "FALHA") + " | " + mensagem)
    if not condicao:
        FALHAS.append(mensagem)


def custo_sequencia(u, r, u_anterior, lamb, rho):
    u = np.asarray(u, float)
    deslocamentos = np.diff(np.concatenate(([u_anterior], u)))
    return float(-(r * u).sum() + lamb * (u ** 2).sum() + rho * (deslocamentos ** 2).sum())


def curva(u, r, caixa):
    return pd.DataFrame({
        "retorno_caixa": caixa, "posicao_u": u,
        "retorno_carteira": r, "capital": np.cumprod(1 + np.asarray(r)),
    })


def dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    feats = obter_lista_features()
    idx = pd.date_range("2022-01-07", periods=n, freq="W-FRI")
    X = rng.normal(0, 1, (n, len(feats)))
    df = pd.DataFrame(X, columns=feats, index=idx)
    df["vix_nivel_k"] = np.abs(rng.normal(20, 5, n))
    df["r_itsa4_k"] = rng.normal(0, 0.03, n)
    df["target"] = df["r_itsa4_k"].shift(-1).fillna(0) * 0.1 + rng.normal(0, 0.02, n)
    return df, feats


def contexto_sintetico(df, feats, retornos_reais, retornos_previstos,
                       vol=0.2, custo=1e-3, caixa=0.0015):
    n = len(df)
    return dict(retornos_reais=retornos_reais, retornos_previstos=retornos_previstos,
                vols_previstas=np.full(n, vol), features=df[feats], datas=df.index,
                custo=custo, retornos_caixa=np.full(n, caixa))


def main():
    rng = np.random.default_rng(1)

    print("\n=== 1. Sharpe de excesso sobre CDI ===")
    n = 120
    ret = rng.normal(0.004, 0.02, n)
    caixa = np.full(n, 0.002)
    res = curva(np.ones(n), ret, caixa)
    m = calcular_metricas(res, 52)
    sharpe_manual = (ret - caixa).mean() / (ret - caixa).std(ddof=1) * np.sqrt(52)
    checar(abs(m["sharpe"] - sharpe_manual) < 1e-9, "Sharpe de excesso == cálculo manual")

    print("\n=== 2. Significância ===")
    bh = curva(np.ones(n), ret, np.zeros(n))
    sig = calcular_significancia(res, bh, 52, 2000, seed=1)
    checar(0 <= sig["p_valor_sharpe_menor_igual_zero"] <= 1, "p-valor Sharpe em [0,1]")
    checar(0 <= sig["p_valor_t"] <= 1, "p-valor t (scipy.stats) em [0,1]")
    checar(sig["sharpe_ic95_inf"] <= sig["sharpe_observado"] <= sig["sharpe_ic95_sup"],
           "Sharpe observado dentro do IC95%")

    print("\n=== 3. Simulador único ===")
    df, feats = dataset(120, 3)
    rr = np.expm1(df["target"].values)
    res_bh = BT.simular(BuyAndHold(), **contexto_sintetico(df, feats, rr, rr, custo=0.0))
    checar(np.allclose(res_bh["posicao_u"], 1.0), "BuyAndHold: u == 1")
    checar(np.allclose(res_bh["capital"].values,
                       np.cumprod(1 + rr)), "capital == cumprod(1+r)")

    print("\n=== 4. resolver_mpc vs força bruta ===")
    ok = True
    for _ in range(15):
        H = int(rng.integers(1, 4))
        r = rng.normal(0, 0.03, H)
        u_ant = float(rng.uniform(0, 1))
        lamb = float(rng.uniform(5e-4, 2e-2))
        rho = float(rng.uniform(1e-3, 1e-1))
        u0 = resolver_mpc(r, u_ant, lamb, rho)
        grade = np.linspace(0, 1, 21)
        melhor = min(itertools.product(grade, repeat=H),
                     key=lambda u: custo_sequencia(u, r, u_ant, lamb, rho))
        if abs(u0 - melhor[0]) > 0.06:
            ok = False
            break
    checar(ok, "resolver_mpc ~ ótimo por força bruta (u_0)")

    print("\n=== 5. MPC(H=1) == baseline ===")
    diferenca = 0.0
    for _ in range(40):
        r = float(rng.normal(0, 0.03))
        u_ant = float(rng.uniform(0, 1))
        lamb = float(rng.uniform(5e-4, 2e-2))
        rho = float(rng.uniform(1e-3, 1e-1))
        u_mpc = resolver_mpc(np.array([r]), u_ant, lamb, rho)
        u_h1 = ControleH1(lamb, rho).alocar(Passo(r, 0.2, None, u_ant))
        analitico = np.clip((r + 2 * rho * u_ant) / (2 * (lamb + rho)), 0, 1)
        diferenca = max(diferenca, abs(u_mpc - u_h1), abs(u_mpc - analitico))
    checar(diferenca < 1e-3, f"MPC(H=1) = baseline = analítico (dif {diferenca:.1e})")

    print("\n=== 6. Previsão multi-passo ===")
    df2, f2 = dataset(160, 4)
    modelo = treinar_modelo(df2[f2], df2["target"], np.ones(len(df2)), 1.0)
    linha = df2[f2].iloc[100]
    for H in [1, 2, 4]:
        checar(len(prever_horizonte(modelo, f2, linha, H)) == H, f"prever_horizonte H={H}")
    p1 = prever_horizonte(modelo, f2, linha, 1)[0]
    checar(abs(p1 - np.expm1(float(modelo.predict(linha[f2].to_frame().T)[0]))) < 1e-12,
           "H=1 == predição direta")

    print("\n=== 7. Walk-forward sem look-ahead ===")
    violacoes = {"n": 0}

    class ModeloEspiao:
        def __init__(self, fim_treino):
            self.fim_treino = fim_treino

        def predict(self, X):
            if X.index.min() <= self.fim_treino:
                violacoes["n"] += 1
            return np.zeros(len(X))

    treinar_espiao = lambda X, y: ModeloEspiao(X.index.max())
    sigma = pd.Series(0.2, index=df.index)
    cdi = pd.Series(0.0015, index=df.index)
    prev = BT.walk_forward_previsoes(df, feats, "target", sigma, cdi, treinar_espiao,
                                     inicio=str(df.index[100].date()), minimo_treino=80)
    checar(violacoes["n"] == 0, "nenhuma previsão usa o futuro")
    checar(len(prev) > 0 and prev.index.is_monotonic_increasing,
           "série OOS crescente e não-vazia")

    print("\n=== 8. Função de transferência ===")
    todos = TF.LAGS_SAIDA + sum(TF.ENTRADAS_EXOGENAS.values(), []) + TF.REGRESSORES_NAO_LTI
    coefs = {f: 0.0 for f in todos}
    coefs["r_itsa4_k"] = 0.5
    coefs["r_itsa4_k_1"] = -0.06
    A, B = TF.extrair_polinomios(coefs)
    polos, _, raio, estavel = TF.calcular_polos_zeros(A, B)
    nao_nulos = np.sort(np.abs(polos[np.abs(polos) > 1e-9]))
    checar(np.allclose([0.2, 0.3], nao_nulos, atol=1e-9) and estavel,
           "polos AR conhecidos + estável")
    coefs2 = {f: 0.0 for f in coefs}
    coefs2["r_itsa4_k"] = 1.5
    _, _, raio2, estavel2 = TF.calcular_polos_zeros(*TF.extrair_polinomios(coefs2))
    checar((not estavel2) and raio2 > 1, "instabilidade detectada")
    coefs3 = {f: 0.0 for f in coefs}
    coefs3["r_itsa4_k"] = 0.4
    coefs3["r_itub4_k"] = 0.2
    coefs3["r_itub4_k_1"] = 0.1
    A3, B3 = TF.extrair_polinomios(coefs3)
    checar(abs(TF.ganho_dc(A3, B3["ITUB4"]) - 0.3 / 0.6) < 1e-12, "ganho DC = B(1)/A(1)")

    print("\n=== 9. Volatility targeting ===")
    nb = 100
    rets = np.concatenate([rng.normal(0, 0.01, nb), rng.normal(0, 0.05, nb)])
    serie = pd.Series(rets, index=pd.date_range("2022-01-07", periods=2 * nb, freq="W-FRI"))
    s = prever_volatilidade(serie, usar_vix=False)
    serie_mutada = serie.copy()
    serie_mutada.iloc[150] += 0.5
    checar(np.allclose(s.values[:150],
                       prever_volatilidade(serie_mutada, usar_vix=False).values[:150]),
           "vol prevista é causal (sem look-ahead)")
    vt = VolTarget(0.15)
    checar(vt.alocar(Passo(0, 0.10, None, 0)) > vt.alocar(Passo(0, 0.50, None, 0)),
           "mais vol -> menos exposição")
    dff, ff = dataset(140, 7)
    rr9 = np.expm1(dff["target"].values)
    res_vt = BT.simular(VolTarget(0.20, 0.1, 0.5), retornos_reais=rr9,
                        retornos_previstos=rr9, vols_previstas=s.values[:len(dff)],
                        features=dff[ff], datas=dff.index, custo=0.001,
                        retornos_caixa=np.full(len(dff), 0.0015))
    checar(res_vt["posicao_u"].between(0, 1).all(), "vol targeting + ARX: u em [0,1]")

    print("\n=== 10. Controlador H=1 e filtros de 1ª ordem ===")
    lamb, rho = 0.01, 0.05
    c = TF.controlador_h1(lamb, rho)
    checar(abs(c["polo"] - rho / (lamb + rho)) < 1e-12 and c["estavel"],
           "polo do controlador = rho/(lamb+rho), estável")
    checar(abs(c["ganho_dc"] - 1 / (2 * lamb)) < 1e-12, "ganho DC = 1/(2*lambda)")
    controlador = ControleH1(lamb, rho)
    u_filtro, u_controle, dif10 = 0.3, 0.3, 0.0
    for r in rng.normal(5e-3, 1e-3, 60):
        u_filtro = c["polo"] * u_filtro + c["ganho_b"] * r
        u_controle = controlador.alocar(Passo(float(r), 0.2, None, u_controle))
        dif10 = max(dif10, abs(u_filtro - u_controle))
    checar(dif10 < 1e-12, f"lei H=1 == filtro u[k] = a u[k-1] + b r (dif {dif10:.1e})")
    f_c, mag_c = TF.resposta_frequencia_controlador(lamb, rho)
    checar(abs(mag_c[0] - c["ganho_dc"]) < 1e-9 and np.all(np.diff(mag_c) <= 1e-12),
           "resposta em frequência: passa-baixa com |C(1)| = ganho DC")
    for a in [0.3, 0.625, 0.9]:
        filtro = TF.filtro_primeira_ordem(a, 1 - a)
        omega3 = 2 * np.pi * filtro["banda_3db"]
        mag3 = abs((1 - a) * np.exp(1j * omega3) / (np.exp(1j * omega3) - a))
        checar(abs(mag3 - filtro["ganho_dc"] / np.sqrt(2)) < 1e-9,
               f"banda de -3 dB correta (a={a})")
    checar(TF.filtro_primeira_ordem(0.0, 1.0)["banda_3db"] == 0.5,
           "a=0: banda = Nyquist")
    ewma = TF.filtro_ewma(0.94)
    checar(abs(ewma["ganho_dc"] - 1.0) < 1e-12 and ewma["polo"] == 0.94,
           "filtro EWMA: polo lambda_v e ganho DC unitário")

    print("\n=== 11. MPC walk-forward ===")
    df11, f11 = dataset(160, 11)
    sigma11 = pd.Series(0.2, index=df11.index)
    cdi11 = pd.Series(0.0015, index=df11.index)
    treinar_real = lambda X, y: treinar_modelo(X, y, np.ones(len(X)), 1.0)
    prev11, modelos11 = BT.walk_forward_previsoes(
        df11, f11, "target", sigma11, cdi11, treinar_real,
        inicio=str(df11.index[100].date()), minimo_treino=80, devolver_modelos=True)
    checar(len(modelos11) == len(prev11), "um modelo por data do walk-forward")
    modelo11 = treinar_real(df11[f11], df11["target"])
    rr11 = np.expm1(df11["target"].values)
    ctx11 = contexto_sintetico(df11, f11, rr11, rr11)
    res_fixo = BT.simular(ControleMPC(modelo11, f11, 0.01, 0.05, 3), **ctx11)
    res_wf11 = BT.simular(ControleMPCWF([modelo11] * len(df11), f11, 0.01, 0.05, 3),
                          **ctx11)
    checar(np.allclose(res_fixo["posicao_u"], res_wf11["posicao_u"], atol=1e-10),
           "ControleMPCWF com modelos idênticos == ControleMPC")

    class ModeloConstante:
        def __init__(self, valor):
            self.valor = valor

        def predict(self, X):
            return np.full(len(X), self.valor)

    n11 = len(df11)
    modelos_a = [ModeloConstante(0.02)] * n11
    modelos_b = [ModeloConstante(0.02)] + [ModeloConstante(-0.02)] * (n11 - 1)
    res_a = BT.simular(ControleMPCWF(modelos_a, f11, 0.01, 0.05, 2), **ctx11)
    res_b = BT.simular(ControleMPCWF(modelos_b, f11, 0.01, 0.05, 2), **ctx11)
    checar(abs(res_a["posicao_u"].iloc[0] - res_b["posicao_u"].iloc[0]) < 1e-9
           and res_b["posicao_u"].iloc[1] < res_a["posicao_u"].iloc[1],
           "ControleMPCWF usa o modelo vigente de cada semana")

    print("\n=== 12. Refino por Nelder-Mead ===")
    df12, f12 = dataset(120, 12)
    rr12 = np.expm1(df12["target"].values)
    ctx12 = contexto_sintetico(df12, f12, rr12, rr12)
    chute = {"lamb": 1e-3, "rho": 1e-2}
    limites12 = {"lamb": (1e-5, 1e-1), "rho": (1e-3, 1e-1)}
    sharpe_chute = calcular_metricas(BT.simular(ControleH1(**chute), **ctx12),
                                     52)["sharpe"]
    params12, sharpe12 = BT.refinar(ControleH1, chute, ctx12, limites12, 52)
    checar(sharpe12 >= sharpe_chute - 1e-9,
           "Nelder-Mead melhora (ou iguala) o chute inicial da grade")
    checar(all(limites12[n][0] <= v <= limites12[n][1] for n, v in params12.items()),
           "parâmetros refinados dentro dos limites da grade")

    print("\n=== 13. Sensibilidade ao atraso ===")
    rr13 = np.tile([0.05, -0.05], 30)
    df13, f13 = dataset(len(rr13), 13)
    ctx13 = contexto_sintetico(df13, f13, rr13, rr13, custo=0.0, caixa=0.0)
    est13 = ControleH1(1e-4, 1e-4)
    capital_sem_atraso = BT.simular(est13, **ctx13)["capital"].iloc[-1]
    capital_com_atraso = BT.simular(
        est13, **BT.atrasar_previsoes(ctx13, 1))["capital"].iloc[-1]
    checar(BT.atrasar_previsoes(ctx13, 0) is ctx13, "atraso 0 preserva o contexto")
    checar(capital_sem_atraso > capital_com_atraso,
           "atraso de 1 semana degrada o desempenho (previsão perfeita)")

    print("\n" + "=" * 50)
    if FALHAS:
        print(f"RESULTADO: {len(FALHAS)} FALHA(S)")
        for falha in FALHAS:
            print("  - " + falha)
        sys.exit(1)
    print("RESULTADO: TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
