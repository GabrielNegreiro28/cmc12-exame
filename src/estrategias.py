"""
Leis de controle para a alocação semanal u[k] em [0, 1] na ITSA4. Todas
implementam o protocolo Estrategia.alocar(Passo) -> u, e um único simulador
(backtest.simular) aplica a mesma dinâmica de capital a qualquer uma delas.

ControleH1 minimiza a cada semana o custo quadrático
J[k] = -r_hat[k+1] u[k] + lambda u[k]^2 + rho (u[k] - u[k-1])^2, cuja solução
analítica é u*[k] = (r_hat[k+1] + 2 rho u[k-1]) / (2 (lambda + rho)),
saturada em [0, 1]; lambda penaliza o esforço de controle e rho a variação da
posição. ControleMPC generaliza para um horizonte de predição H: prevê
r_hat[k+1..k+H] pela recursão do estado do ARX e minimiza
J = soma_j (-r_hat u + lambda u^2 + rho (Delta u)^2) sujeito a 0 <= u <= 1,
aplicando apenas o primeiro movimento (horizonte deslizante). ControleMPCWF é
a variante para o walk-forward, que usa em cada semana o modelo
re-identificado vigente naquela data. VolTarget regula a exposição por
sigma*/sigma_hat[k], com suavização e tilt opcional pelo sinal do ARX, e
BuyAndHold é a referência u = 1.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LAGS_ITSA4 = ["r_itsa4_k", "r_itsa4_k_1", "r_itsa4_k_2", "r_itsa4_k_3"]


@dataclass(frozen=True)
class Passo:
    retorno_previsto: float
    vol_prevista: float
    features: pd.Series
    u_anterior: float


class Estrategia(Protocol):
    nome: str

    def alocar(self, passo: Passo) -> float: ...


def _saturar(u: float) -> float:
    return float(min(max(u, 0.0), 1.0))


@dataclass(frozen=True)
class ControleH1:
    lamb: float
    rho: float
    nome: str = "Controle H=1"

    def alocar(self, p: Passo) -> float:
        u = (p.retorno_previsto + 2 * self.rho * p.u_anterior) / (2 * (self.lamb + self.rho))
        return _saturar(u)


def _avancar_estado(linha: pd.Series, r_log_previsto: float) -> pd.Series:
    nova = linha.copy()
    nova["r_itsa4_k_3"] = linha["r_itsa4_k_2"]
    nova["r_itsa4_k_2"] = linha["r_itsa4_k_1"]
    nova["r_itsa4_k_1"] = linha["r_itsa4_k"]
    nova["r_itsa4_k"] = r_log_previsto
    return nova


def prever_horizonte(modelo, features, linha_atual, horizonte) -> np.ndarray:
    linha = linha_atual.copy()
    previsoes_log = []
    for _ in range(horizonte):
        previsao = float(modelo.predict(linha[features].to_frame().T)[0])
        previsoes_log.append(previsao)
        linha = _avancar_estado(linha, previsao)
    return np.expm1(np.array(previsoes_log))


def resolver_mpc(r_hat: np.ndarray, u_anterior: float, lamb: float, rho: float) -> float:
    H = len(r_hat)

    def J(u):
        deslocamentos = np.diff(np.concatenate(([u_anterior], u)))
        return float(-(r_hat * u).sum() + lamb * (u ** 2).sum()
                     + rho * (deslocamentos ** 2).sum())

    def gradiente_J(u):
        g = -r_hat + 2 * lamb * u
        g += 2 * rho * (u - np.concatenate(([u_anterior], u[:-1])))
        g[:-1] -= 2 * rho * (u[1:] - u[:-1])
        return g

    u0 = np.full(H, _saturar(u_anterior))
    solucao = minimize(J, u0, jac=gradiente_J, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * H)
    return _saturar(float(solucao.x[0]))


@dataclass(frozen=True)
class ControleMPC:
    modelo: object
    features: list
    lamb: float
    rho: float
    horizonte: int
    nome: str = "MPC"

    def alocar(self, p: Passo) -> float:
        r_hat = prever_horizonte(self.modelo, self.features, p.features, self.horizonte)
        return resolver_mpc(r_hat, p.u_anterior, self.lamb, self.rho)


class ControleMPCWF:
    def __init__(self, modelos: list, features: list, lamb: float, rho: float,
                 horizonte: int):
        self.modelos = modelos
        self.features = features
        self.lamb = lamb
        self.rho = rho
        self.horizonte = horizonte
        self.nome = "MPC (WF)"
        self._k = 0

    def alocar(self, p: Passo) -> float:
        modelo = self.modelos[min(self._k, len(self.modelos) - 1)]
        self._k += 1
        r_hat = prever_horizonte(modelo, self.features, p.features, self.horizonte)
        return resolver_mpc(r_hat, p.u_anterior, self.lamb, self.rho)


@dataclass(frozen=True)
class VolTarget:
    sigma_alvo: float
    rho_suav: float = 0.10
    ganho_tilt: float = 0.0
    nome: str = "Volatility targeting"

    def alocar(self, p: Passo) -> float:
        u_ref = self.sigma_alvo / max(p.vol_prevista, 1e-8)
        if self.ganho_tilt > 0:
            vol_semanal = max(p.vol_prevista / np.sqrt(52), 1e-6)
            tilt = 1.0 + self.ganho_tilt * np.tanh(p.retorno_previsto / vol_semanal)
            u_ref *= min(max(tilt, 0.0), 2.0)
        u = (u_ref + self.rho_suav * p.u_anterior) / (1.0 + self.rho_suav)
        return _saturar(u)


@dataclass(frozen=True)
class BuyAndHold:
    nome: str = "Buy and Hold"

    def alocar(self, p: Passo) -> float:
        return 1.0
