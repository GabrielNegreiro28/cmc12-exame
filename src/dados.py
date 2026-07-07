"""
Camada de dados: download das séries de mercado (Yahoo Finance) e do CDI
(SGS/BCB), snapshot em disco para reprodutibilidade, conversão para a
frequência semanal e cálculo dos retornos logarítmicos. A função de alto
nível preparar_mercado devolve um DadosMercado imutável; na primeira execução
os dados são baixados e salvos em snapshot, e nas seguintes são recarregados
do disco, de modo que duas execuções produzem o mesmo resultado. As
bibliotecas de rede (yfinance e requests) são importadas apenas quando o
download é necessário, permitindo execução totalmente offline a partir do
snapshot.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config


@dataclass(frozen=True)
class DadosMercado:
    dados_semanais: pd.DataFrame
    retornos: pd.DataFrame
    cdi_semanal: pd.Series


def baixar_dados_mercado(tickers, data_inicial, data_final) -> pd.DataFrame:
    import yfinance as yf
    print("Baixando dados de mercado...")
    return yf.download(
        list(tickers.values()), start=data_inicial, end=data_final,
        auto_adjust=True, progress=False,
    )


def preparar_precos_volume(dados, tickers):
    precos = dados["Close"].copy()
    volume = dados["Volume"].copy()

    esperados = list(tickers.values())
    faltando = [t for t in esperados if t not in precos.columns]
    if faltando:
        raise ValueError(
            f"Download incompleto. Faltando: {faltando}. "
            f"Recebido: {list(precos.columns)}."
        )

    mapa = {v: k for k, v in tickers.items()}
    precos = precos.rename(columns=mapa)[list(tickers)]
    volume = volume.rename(columns=mapa)
    return precos, volume["ITSA4"].copy()


def converter_para_semanal(precos, volume_itsa4, frequencia):
    semanais = precos.resample(frequencia).last()
    semanais["Volume_ITSA4"] = volume_itsa4.resample(frequencia).sum()
    return semanais.dropna()


def calcular_retornos_log(dados_semanais, ativos):
    r = np.log(dados_semanais[ativos] / dados_semanais[ativos].shift(1))
    return r.dropna()


def baixar_serie_bcb(codigo, data_inicial, data_final) -> pd.Series:
    import requests
    inicio, fim = pd.to_datetime(data_inicial), pd.to_datetime(data_final)
    partes, bloco_inicio = [], inicio
    while bloco_inicio <= fim:
        bloco_fim = min(bloco_inicio + pd.DateOffset(years=2), fim)
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
            f"?formato=json&dataInicial={bloco_inicio:%d/%m/%Y}"
            f"&dataFinal={bloco_fim:%d/%m/%Y}"
        )
        for tentativa in range(3):
            try:
                resposta = requests.get(
                    url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
                resposta.raise_for_status()
                registros = resposta.json()
                if registros:
                    s = pd.DataFrame(registros)
                    s["data"] = pd.to_datetime(s["data"], dayfirst=True)
                    s["valor"] = pd.to_numeric(
                        s["valor"].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce")
                    partes.append(s.set_index("data")["valor"])
                break
            except Exception as erro:
                print(f"Falha BCB (tentativa {tentativa + 1}): {erro}")
                time.sleep(5)
        else:
            raise RuntimeError(f"Não foi possível baixar a série {codigo}.")
        bloco_inicio = bloco_fim + pd.DateOffset(days=1)

    serie = pd.concat(partes)
    return serie[~serie.index.duplicated()].sort_index()


def calcular_retorno_cdi_semanal(cdi_diario_percentual, indice_semanal, frequencia):
    fator = (1 + cdi_diario_percentual / 100.0).resample(frequencia).prod()
    retorno = (fator - 1).reindex(indice_semanal)
    return retorno.ffill().fillna(0.0)


def _salvar(df, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho)
    print(f"Snapshot salvo: {caminho}")


def preparar_mercado(cfg: Config, snapshot_dir: str) -> DadosMercado:
    ativos = list(cfg.tickers)
    snap_precos = os.path.join(snapshot_dir, "precos_diarios.csv")
    snap_volume = os.path.join(snapshot_dir, "volume_itsa4_diario.csv")
    snap_cdi = os.path.join(snapshot_dir, "cdi_diario.csv")

    if cfg.usar_snapshot and all(os.path.exists(p) for p in (snap_precos, snap_volume, snap_cdi)):
        print("\nCarregando dados do snapshot (modo reprodutível)...")
        precos = pd.read_csv(snap_precos, index_col=0, parse_dates=True)
        volume = pd.read_csv(snap_volume, index_col=0, parse_dates=True).iloc[:, 0]
        cdi_diario = pd.read_csv(snap_cdi, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        brutos = baixar_dados_mercado(cfg.tickers, cfg.data_inicial, cfg.data_final)
        precos, volume = preparar_precos_volume(brutos, cfg.tickers)
        print("\nBaixando CDI (SGS/BCB)...")
        cdi_diario = baixar_serie_bcb(cfg.codigo_cdi_sgs, cfg.data_inicial, cfg.data_final)
        _salvar(precos, snap_precos)
        _salvar(volume.to_frame("Volume_ITSA4"), snap_volume)
        _salvar(cdi_diario.to_frame("cdi_diario"), snap_cdi)

    dados_semanais = converter_para_semanal(precos, volume, cfg.frequencia)
    retornos = calcular_retornos_log(dados_semanais, ativos)
    cdi_semanal = calcular_retorno_cdi_semanal(cdi_diario, retornos.index, cfg.frequencia)
    return DadosMercado(dados_semanais, retornos, cdi_semanal)
