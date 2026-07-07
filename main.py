"""
Orquestrador do experimento: identificação ARX e controle ótimo discreto para
a alocação semanal em ITSA4. Fluxo executado: preparação dos dados (snapshot
reprodutível); identificação do ARX por ridge; análise do sistema por função
de transferência G(z) = B(z)/A(z), incluindo a sensibilidade dos polos ao
alpha do ridge; seleção de hiperparâmetros do controlador na validação por
busca em grade seguida de refino por Nelder-Mead (com o ótimo da grade como
chute inicial, conforme o tutorial de otimização de CMC-12); caracterização
do controlador C(z) e do estimador EWMA como filtros discretos de primeira
ordem (polo, ganho DC, constante de tempo e banda passante de -3 dB);
avaliação das estratégias no conjunto de teste e na validação walk-forward
com re-identificação do modelo, ambas com significância por bootstrap;
sensibilidade ao custo de transação e ao atraso da previsão (atraso de
transporte); qualidade do previsor contra o previsor nulo; e geração das
saídas, com tabelas CSV em outputs/ e figuras em figuras/.

Execução: py main.py
"""

import itertools
import os
import re
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURAS_DIR = os.path.join(BASE_DIR, "figuras")

from config import Config
import backtest as BT
import dados as D
import features as F
import funcao_transferencia as TF
import identificacao as ID
import metricas as M
import relatorio as R
from estrategias import BuyAndHold, ControleH1, ControleMPC, ControleMPCWF, VolTarget


def slug(nome: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")


def montar_contexto(conjunto, modelo, feats, sigma, cdi_semanal, custo) -> dict:
    X, y = conjunto[feats], conjunto["target"]
    return dict(
        retornos_reais=np.expm1(y.values),
        retornos_previstos=np.expm1(modelo.predict(X)),
        vols_previstas=sigma.reindex(y.index).ffill().bfill().values,
        features=X, datas=y.index, custo=custo,
        retornos_caixa=cdi_semanal.reindex(y.index).ffill().fillna(0.0).values,
    )


def selecionar_hiperparametros(cfg, ctx_validacao):
    grade = [{"lamb": lamb, "rho": rho}
             for lamb, rho in itertools.product(cfg.lista_lamb, cfg.lista_rho)]
    tabela = BT.varrer(ControleH1, grade, ctx_validacao, cfg.periodos_por_ano)
    melhor = tabela.loc[tabela["sharpe"].idxmax()]
    chute_inicial = {"lamb": float(melhor["lamb"]), "rho": float(melhor["rho"])}
    limites = {"lamb": (min(cfg.lista_lamb), max(cfg.lista_lamb)),
               "rho": (min(cfg.lista_rho), max(cfg.lista_rho))}
    refinado, sharpe_refinado = BT.refinar(ControleH1, chute_inicial, ctx_validacao,
                                           limites, cfg.periodos_por_ano)
    historico = pd.DataFrame([
        {"metodo": "grade", **chute_inicial, "sharpe_validacao": float(melhor["sharpe"])},
        {"metodo": "nelder_mead", **refinado, "sharpe_validacao": sharpe_refinado},
    ])
    return refinado, historico


def avaliar_estrategias(estrategias, ctx, periodos_por_ano, benchmark):
    resultados = {nome: BT.simular(est, **ctx) for nome, est in estrategias.items()}
    linhas = []
    for nome, res in resultados.items():
        significancia = M.calcular_significancia(res, resultados[benchmark],
                                                 periodos_por_ano)
        linhas.append({"estrategia": nome,
                       **M.calcular_metricas(res, periodos_por_ano),
                       "p_supera_benchmark":
                           significancia["p_valor_nao_superar_benchmark"]})
    return resultados, pd.DataFrame(linhas)


def sensibilidade_custos(estrategias, ctx, lista_custo, periodos_por_ano) -> pd.DataFrame:
    linhas = []
    for custo in lista_custo:
        ctx_custo = dict(ctx, custo=custo)
        for nome, est in estrategias.items():
            met = M.calcular_metricas(BT.simular(est, **ctx_custo), periodos_por_ano)
            linhas.append({"custo": custo, "estrategia": nome,
                           "sharpe": met["sharpe"],
                           "retorno_total": met["retorno_total"],
                           "troca_media": met["troca_media"]})
    return pd.DataFrame(linhas)


def sensibilidade_atraso(estrategias, ctx, lista_atraso, periodos_por_ano) -> pd.DataFrame:
    linhas = []
    for atraso in lista_atraso:
        ctx_atrasado = BT.atrasar_previsoes(ctx, atraso)
        for nome, est in estrategias.items():
            met = M.calcular_metricas(BT.simular(est, **ctx_atrasado), periodos_por_ano)
            linhas.append({"atraso_semanas": atraso, "estrategia": nome,
                           "sharpe": met["sharpe"],
                           "retorno_total": met["retorno_total"]})
    return pd.DataFrame(linhas)


def sensibilidade_alpha_polos(retreino, feats, pesos_de, lista_alpha) -> pd.DataFrame:
    linhas = []
    for alpha in lista_alpha:
        modelo = ID.treinar_modelo(retreino[feats], retreino["target"],
                                   pesos_de(retreino), alpha)
        analise = TF.analisar(modelo, feats)
        linhas.append({"alpha_ridge": alpha,
                       "raio_espectral": analise["raio_espectral"],
                       "estavel": analise["estavel"]})
    return pd.DataFrame(linhas)


def qualidade_previsor(teste, feats, modelo, previsoes_wf) -> pd.DataFrame:
    real_teste = np.expm1(teste["target"].values)
    return pd.DataFrame([
        {"previsor": "ARX ridge (teste)",
         **M.avaliar_previsor(real_teste, np.expm1(modelo.predict(teste[feats])))},
        {"previsor": "nulo r_hat=0 (teste)",
         **M.avaliar_previsor(real_teste, np.zeros(len(teste)))},
        {"previsor": "ARX ridge (walk-forward)",
         **M.avaliar_previsor(previsoes_wf["retorno_real"],
                              previsoes_wf["retorno_previsto"])},
        {"previsor": "nulo r_hat=0 (walk-forward)",
         **M.avaliar_previsor(previsoes_wf["retorno_real"],
                              np.zeros(len(previsoes_wf)))},
    ])


def main():
    cfg = Config()
    ppa = cfg.periodos_por_ano
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURAS_DIR, exist_ok=True)

    mkt = D.preparar_mercado(cfg, os.path.join(OUTPUT_DIR, "snapshot"))
    df = F.criar_features_arx(mkt.retornos, mkt.dados_semanais, mkt.cdi_semanal)
    feats = F.obter_lista_features()
    sigma = F.prever_volatilidade(df["r_itsa4_k"], df["vix_nivel_k"],
                                  cfg.lambda_ewma_vol, periodos_por_ano=ppa)

    treino = df.loc[cfg.inicio_treino:cfg.fim_treino]
    validacao = df.loc[cfg.inicio_validacao:cfg.fim_validacao]
    retreino = df.loc[cfg.inicio_retreino:cfg.fim_retreino]
    teste = df.loc[cfg.inicio_teste:]
    for nome, parte in [("treino", treino), ("validação", validacao), ("teste", teste)]:
        if parte.empty:
            raise ValueError(f"Conjunto '{nome}' vazio: verifique datas e snapshot.")

    def pesos_de(parte):
        if not cfg.usar_pesos_temporais:
            return np.ones(len(parte))
        return ID.criar_pesos_temporais(parte.index, cfg.anos_recentes_peso,
                                        cfg.peso_recente)

    def treinar(X, y):
        return ID.treinar_modelo(X, y, pesos_de(X), cfg.alpha_ridge)

    modelo_validacao = treinar(treino[feats], treino["target"])
    modelo_teste = treinar(retreino[feats], retreino["target"])
    R.salvar_csv(ID.obter_coeficientes(modelo_teste, feats), OUTPUT_DIR,
                 "coeficientes_arx.csv", indice=False)

    analise_tf = TF.analisar(modelo_teste, feats)
    print(f"\nG(z): raio espectral = {analise_tf['raio_espectral']:.4f} | "
          f"estável? {analise_tf['estavel']}")
    R.salvar_csv(TF.tabela_polos(analise_tf), OUTPUT_DIR,
                 "polos_funcao_transferencia.csv", indice=False)
    R.salvar_csv(sensibilidade_alpha_polos(retreino, feats, pesos_de,
                                           cfg.lista_alpha_ridge),
                 OUTPUT_DIR, "sensibilidade_alpha_polos.csv", indice=False)

    ctx_validacao = montar_contexto(validacao, modelo_validacao, feats, sigma,
                                    mkt.cdi_semanal, cfg.custo_transacao)
    hiper, historico_hiper = selecionar_hiperparametros(cfg, ctx_validacao)
    lamb, rho = hiper["lamb"], hiper["rho"]
    tabela_h = BT.varrer(
        lambda horizonte: ControleMPC(modelo_validacao, feats, lamb, rho, horizonte),
        [{"horizonte": h} for h in cfg.lista_horizonte], ctx_validacao, ppa)
    horizonte = int(tabela_h.loc[tabela_h["sharpe"].idxmax(), "horizonte"])
    R.salvar_csv(tabela_h[["horizonte", "sharpe", "retorno_total",
                           "volatilidade_anualizada", "troca_media"]],
                 OUTPUT_DIR, "selecao_horizonte.csv", indice=False)
    R.salvar_csv(historico_hiper, OUTPUT_DIR, "hiperparametros.csv", indice=False)
    print(f"Hiperparâmetros: lambda={lamb:.6f}, rho={rho:.6f}, H={horizonte}")

    filtros = pd.DataFrame([
        {"filtro": "Controlador C(z), H=1", **TF.controlador_h1(lamb, rho)},
        {"filtro": "EWMA da variância", **TF.filtro_ewma(cfg.lambda_ewma_vol)},
    ]).drop(columns="ganho_b")
    R.salvar_csv(filtros, OUTPUT_DIR, "filtros_primeira_ordem.csv", indice=False)

    estrategias = {
        "Controle H=1": ControleH1(lamb, rho),
        f"MPC (H={horizonte})": ControleMPC(modelo_teste, feats, lamb, rho, horizonte),
        "Volatility targeting": VolTarget(cfg.sigma_alvo, cfg.rho_suavizacao),
        "Vol targeting + ARX": VolTarget(cfg.sigma_alvo, cfg.rho_suavizacao,
                                         cfg.ganho_tilt),
        "Buy and Hold": BuyAndHold(),
    }
    ctx_teste = montar_contexto(teste, modelo_teste, feats, sigma,
                                mkt.cdi_semanal, cfg.custo_transacao)
    resultados, resumo = avaliar_estrategias(estrategias, ctx_teste, ppa,
                                             "Buy and Hold")
    R.salvar_csv(resumo, OUTPUT_DIR, "resumo_estrategias_teste.csv", indice=False)
    for nome, res in resultados.items():
        R.salvar_csv(res, OUTPUT_DIR, f"resultado_{slug(nome)}_teste.csv")

    R.salvar_csv(sensibilidade_custos(estrategias, ctx_teste, cfg.lista_custo, ppa),
                 OUTPUT_DIR, "sensibilidade_custos.csv", indice=False)
    preditivas = {nome: estrategias[nome]
                  for nome in ["Controle H=1", f"MPC (H={horizonte})"]}
    if horizonte != 4:
        preditivas["MPC (H=4)"] = ControleMPC(modelo_teste, feats, lamb, rho, 4)
    R.salvar_csv(sensibilidade_atraso(preditivas, ctx_teste, cfg.lista_atraso, ppa),
                 OUTPUT_DIR, "sensibilidade_atraso.csv", indice=False)

    previsoes_wf, modelos_wf = BT.walk_forward_previsoes(
        df, feats, "target", sigma, mkt.cdi_semanal, treinar,
        cfg.inicio_walkforward, devolver_modelos=True)
    ctx_wf = dict(
        retornos_reais=previsoes_wf["retorno_real"].values,
        retornos_previstos=previsoes_wf["retorno_previsto"].values,
        vols_previstas=previsoes_wf["vol_prevista"].values,
        features=df[feats].reindex(previsoes_wf.index), datas=previsoes_wf.index,
        custo=cfg.custo_transacao, retornos_caixa=previsoes_wf["cdi"].values)
    estrategias_wf = {
        "Controle H=1 (WF)": ControleH1(lamb, rho),
        f"MPC (H={horizonte}, WF)": ControleMPCWF(modelos_wf, feats, lamb, rho,
                                                  horizonte),
        "Volatility targeting (WF)": VolTarget(cfg.sigma_alvo, cfg.rho_suavizacao),
        "Buy and Hold (WF)": BuyAndHold(),
    }
    resultados_wf, resumo_wf = avaliar_estrategias(estrategias_wf, ctx_wf, ppa,
                                                   "Buy and Hold (WF)")
    R.salvar_csv(resumo_wf, OUTPUT_DIR, "resumo_walkforward.csv", indice=False)

    R.salvar_csv(qualidade_previsor(teste, feats, modelo_teste, previsoes_wf),
                 OUTPUT_DIR, "qualidade_previsor.csv", indice=False)

    R.grafico_capital(resultados, FIGURAS_DIR, "capital_estrategias_teste.png",
                      "Teste: comparação das estratégias")
    R.grafico_capital(resultados_wf, FIGURAS_DIR, "capital_walkforward.png",
                      "Walk-forward: controle vs vol targeting vs Buy and Hold")
    R.grafico_polos_zeros(analise_tf, FIGURAS_DIR)
    R.grafico_resposta_frequencia(analise_tf, FIGURAS_DIR)
    R.grafico_resposta_controlador(lamb, rho, FIGURAS_DIR)
    R.grafico_exposicao_vol(resultados["Volatility targeting"],
                            ctx_teste["vols_previstas"], ctx_teste["datas"],
                            cfg.sigma_alvo, FIGURAS_DIR)

    print(f"\nExecução finalizada. Tabelas em {OUTPUT_DIR}, figuras em {FIGURAS_DIR}")


if __name__ == "__main__":
    main()
