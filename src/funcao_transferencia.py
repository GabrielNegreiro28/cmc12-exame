"""
Análise no domínio z do sistema identificado e dos filtros do projeto. Do ARX
extrai-se a função de transferência de pulso G(z) = B(z)/A(z): o denominador
A(z) vem da parte autorregressiva da ITSA4 (seus polos determinam a
estabilidade, que exige |z_i| < 1, e a memória da série) e os numeradores
B_i(z) vêm das entradas exógenas; os regressores não lineares (volatilidades,
tendências, volume, níveis de VIX/CDI) ficam fora da decomposição LTI. O
módulo também caracteriza filtros discretos de primeira ordem da forma
H(z) = b z / (z - a): o controlador de horizonte 1, cuja lei não saturada
u[k] = a u[k-1] + b r_hat[k+1] tem C(z) = U(z)/R_hat(z) = b z / (z - a) com
a = rho/(lambda+rho) e ganho DC 1/(2 lambda), e o estimador EWMA da
variância, que é o mesmo filtro com polo lambda_v e ganho DC unitário. Para
ambos calculam-se polo, ganho DC, constante de tempo e banda passante de
-3 dB, ligando os hiperparâmetros do custo quadrático à resposta em
frequência.
"""

import numpy as np
import pandas as pd

LAGS_SAIDA = ["r_itsa4_k", "r_itsa4_k_1", "r_itsa4_k_2", "r_itsa4_k_3"]

ENTRADAS_EXOGENAS = {
    "ITUB4": ["r_itub4_k", "r_itub4_k_1", "r_itub4_k_2"],
    "IBOV": ["r_ibov_k", "r_ibov_k_1", "r_ibov_k_2"],
    "USDBRL": ["r_usdbrl_k", "r_usdbrl_k_1"],
    "EWZ": ["r_ewz_k", "r_ewz_k_1"],
    "VIX": ["r_vix_k", "r_vix_k_1"],
}

REGRESSORES_NAO_LTI = [
    "vol_itsa4_4s", "vol_itsa4_12s", "volume_relativo_itsa4",
    "tendencia_itsa4_12s", "vol_ibov_4s", "tendencia_ibov_12s",
    "vix_nivel_k", "nivel_cdi_k",
]


def coeficientes_brutos(modelo_ridge, features) -> dict:
    ridge = modelo_ridge.named_steps["ridge"]
    escala = modelo_ridge.named_steps["scaler"].scale_
    return {f: float(c / s) for f, c, s in zip(features, ridge.coef_, escala)}


def extrair_polinomios(coefs):
    a = [coefs.get(f, 0.0) for f in LAGS_SAIDA]
    A_z = np.array([1.0, -a[0], -a[1], -a[2], -a[3]])
    B_z = {}
    for entrada, feats in ENTRADAS_EXOGENAS.items():
        num = np.zeros(5)
        for i, f in enumerate(feats):
            num[i + 1] = coefs.get(f, 0.0)
        B_z[entrada] = num
    return A_z, B_z


def calcular_polos_zeros(A_z, B_z):
    polos = np.roots(A_z)
    zeros = {e: (np.roots(np.trim_zeros(n, "f")) if len(np.trim_zeros(n, "f")) > 1
                 else np.array([]))
             for e, n in B_z.items()}
    raio = float(np.max(np.abs(polos))) if len(polos) else 0.0
    return polos, zeros, raio, raio < 1.0


def ganho_dc(A_z, B_num):
    den = np.polyval(A_z, 1.0)
    return np.nan if den == 0 else np.polyval(B_num, 1.0) / den


def resposta_frequencia(A_z, B_num, n_pontos=512):
    omega = np.linspace(0, np.pi, n_pontos)
    z = np.exp(1j * omega)
    H = np.polyval(B_num, z) / np.polyval(A_z, z)
    return omega / (2 * np.pi), np.abs(H)


def analisar(modelo_ridge, features) -> dict:
    coefs = coeficientes_brutos(modelo_ridge, features)
    A_z, B_z = extrair_polinomios(coefs)
    polos, zeros, raio, estavel = calcular_polos_zeros(A_z, B_z)
    return {
        "A_z": A_z, "B_z": B_z, "polos": polos, "zeros": zeros,
        "raio_espectral": raio, "estavel": estavel,
        "ganhos_dc": {e: ganho_dc(A_z, B_z[e]) for e in B_z},
    }


def tabela_polos(analise) -> pd.DataFrame:
    linhas = [{"polo_real": p.real, "polo_imag": p.imag, "modulo": abs(p)}
              for p in analise["polos"]]
    return pd.DataFrame(linhas)


def banda_passante_3db(a: float) -> float:
    if a <= 0.0:
        return 0.5
    cos_omega = (-1.0 + 4.0 * a - a * a) / (2.0 * a)
    omega = np.arccos(np.clip(cos_omega, -1.0, 1.0))
    return float(omega / (2.0 * np.pi))


def filtro_primeira_ordem(a: float, b: float) -> dict:
    return {
        "polo": a,
        "ganho_b": b,
        "ganho_dc": b / (1.0 - a) if a < 1.0 else np.inf,
        "constante_tempo": -1.0 / np.log(a) if 0.0 < a < 1.0 else 0.0,
        "banda_3db": banda_passante_3db(a),
        "estavel": abs(a) < 1.0,
    }


def controlador_h1(lamb: float, rho: float) -> dict:
    return filtro_primeira_ordem(rho / (lamb + rho), 1.0 / (2.0 * (lamb + rho)))


def filtro_ewma(lambda_ewma: float) -> dict:
    return filtro_primeira_ordem(lambda_ewma, 1.0 - lambda_ewma)


def resposta_frequencia_controlador(lamb, rho, n_pontos=512):
    c = controlador_h1(lamb, rho)
    omega = np.linspace(0, np.pi, n_pontos)
    z = np.exp(1j * omega)
    H = c["ganho_b"] * z / (z - c["polo"])
    return omega / (2 * np.pi), np.abs(H)
