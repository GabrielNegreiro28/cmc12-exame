"""
Configuração única e imutável do experimento. Concentra todas as constantes:
ativos, datas da separação temporal, hiperparâmetros de identificação e das
leis de controle, grades de busca e custos. A separação temporal é desenhada
para que nenhuma janela de seleção se sobreponha a uma janela de avaliação:
o modelo inicial é identificado no treino (2022-2023), os hiperparâmetros do
controlador são escolhidos na validação (2024), o modelo final é re-treinado
até o fim de 2025 com folga antes do teste (2026), e o walk-forward (2025 em
diante) permanece fora da amostra tanto para o modelo quanto para os
hiperparâmetros.
"""

from dataclasses import dataclass, field
from typing import Dict, List


def _tickers_padrao() -> Dict[str, str]:
    return {
        "ITSA4": "ITSA4.SA",
        "IBOV": "^BVSP",
        "USDBRL": "USDBRL=X",
        "ITUB4": "ITUB4.SA",
        "EWZ": "EWZ",
        "VIX": "^VIX",
    }


@dataclass(frozen=True)
class Config:
    tickers: Dict[str, str] = field(default_factory=_tickers_padrao)
    data_inicial: str = "2022-01-01"
    data_final: str = "2026-07-01"
    frequencia: str = "W-FRI"
    periodos_por_ano: int = 52
    codigo_cdi_sgs: int = 12
    usar_snapshot: bool = True

    alpha_ridge: float = 1.0
    lista_alpha_ridge: List[float] = field(default_factory=lambda: [
        0.1, 0.3, 1.0, 3.0, 10.0])
    usar_pesos_temporais: bool = True
    anos_recentes_peso: int = 1
    peso_recente: float = 3.0

    inicio_treino: str = "2022-01-01"
    fim_treino: str = "2023-12-31"
    inicio_validacao: str = "2024-01-01"
    fim_validacao: str = "2024-12-31"
    inicio_retreino: str = "2023-01-01"
    fim_retreino: str = "2025-12-20"
    inicio_teste: str = "2026-01-01"
    inicio_walkforward: str = "2025-01-01"

    lista_lamb: List[float] = field(default_factory=lambda: [
        1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1])
    lista_rho: List[float] = field(default_factory=lambda: [
        1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1])
    lista_horizonte: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    custo_transacao: float = 1e-3
    lista_custo: List[float] = field(default_factory=lambda: [
        0.0, 5e-4, 1e-3, 2e-3, 5e-3])
    lista_atraso: List[int] = field(default_factory=lambda: [0, 1, 2])

    lista_sigma_alvo: List[float] = field(default_factory=lambda: [
        0.10, 0.15, 0.20, 0.25, 0.30])
    sigma_alvo: float = 0.15
    rho_suavizacao: float = 0.10
    ganho_tilt: float = 0.5
    lambda_ewma_vol: float = 0.94
