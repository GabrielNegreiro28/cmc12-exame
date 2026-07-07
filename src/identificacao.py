"""
Identificação do sistema pelo arcabouço de erro de previsão: o retorno log
semanal da ITSA4 é descrito pelo preditor de um passo r[k+1] = phi[k]' theta
+ e[k+1], em que phi[k] reúne os regressores disponíveis no instante k. Os
parâmetros theta são estimados por regressão ridge (penalização L2, adequada
à multicolinearidade dos regressores) sobre variáveis padronizadas, com pesos
amostrais que privilegiam o passado recente.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def criar_pesos_temporais(indice, anos_recentes=1, peso_recente=3.0) -> np.ndarray:
    indice = pd.to_datetime(indice)
    corte = indice.max() - pd.DateOffset(years=anos_recentes)
    pesos = np.ones(len(indice))
    pesos[indice >= corte] = peso_recente
    return pesos


def treinar_modelo(X, y, pesos, alpha_ridge=1.0) -> Pipeline:
    modelo = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha_ridge))])
    modelo.fit(X, y, ridge__sample_weight=pesos)
    return modelo


def obter_coeficientes(modelo, features) -> pd.DataFrame:
    coefs = pd.DataFrame({
        "regressor": features,
        "coeficiente": modelo.named_steps["ridge"].coef_,
    })
    ordem = coefs["coeficiente"].abs().sort_values(ascending=False).index
    return coefs.loc[ordem]
