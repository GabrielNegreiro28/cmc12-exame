"""
Geração de saídas: tabelas em CSV e figuras em PNG. Isola toda a apresentação
(matplotlib) do resto do projeto — nenhum outro módulo desenha gráficos — e
recebe o diretório de destino por parâmetro, de modo que as tabelas vão para
outputs/ e as figuras diretamente para figuras/, a pasta referenciada pelo
relatório em LaTeX (fonte única, sem cópias manuais).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from funcao_transferencia import (controlador_h1, resposta_frequencia,
                                  resposta_frequencia_controlador)


def salvar_csv(df, output_dir, nome, indice=True):
    caminho = os.path.join(output_dir, nome)
    df.to_csv(caminho, index=indice)
    print(f"Arquivo salvo: {caminho}")


def _salvar_figura(output_dir, nome):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, nome), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura salva: {nome}")


def grafico_capital(resultados, output_dir, nome, titulo):
    plt.figure(figsize=(12, 5))
    for rotulo, res in resultados.items():
        plt.plot(res.index, res["capital"], label=rotulo, linewidth=1.5)
    plt.xlabel("Data"); plt.ylabel("Capital normalizado")
    plt.title(titulo); plt.grid(True); plt.legend()
    _salvar_figura(output_dir, nome)


def grafico_polos_zeros(analise, output_dir):
    plt.figure(figsize=(6, 6))
    t = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(t), np.sin(t), color="black", linewidth=0.8)
    polos = analise["polos"]
    plt.scatter(polos.real, polos.imag, marker="x", s=90, color="red", label="Polos")
    for zeros in analise["zeros"].values():
        if len(zeros):
            plt.scatter(zeros.real, zeros.imag, marker="o", facecolors="none",
                        edgecolors="C0", s=60)
    plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("Re"); plt.ylabel("Im")
    plt.title(f"Polos e zeros (raio espectral = {analise['raio_espectral']:.3f})")
    plt.legend(); plt.grid(True)
    _salvar_figura(output_dir, "mapa_polos_zeros_arx.png")


def grafico_resposta_frequencia(analise, output_dir):
    plt.figure(figsize=(10, 5))
    for entrada, num in analise["B_z"].items():
        f, mag = resposta_frequencia(analise["A_z"], num)
        plt.plot(f, 20 * np.log10(np.maximum(mag, 1e-12)), label=entrada)
    plt.xlabel("Frequência (ciclos/semana)"); plt.ylabel("Magnitude (dB)")
    plt.title("Resposta em frequência do ARX por entrada")
    plt.grid(True); plt.legend()
    _salvar_figura(output_dir, "resposta_frequencia_arx.png")


def grafico_resposta_controlador(lamb, rho, output_dir):
    plt.figure(figsize=(10, 5))
    for fator, estilo in [(0.1, "--"), (1.0, "-"), (10.0, ":")]:
        r = rho * fator
        f, mag = resposta_frequencia_controlador(lamb, r)
        polo = controlador_h1(lamb, r)["polo"]
        rotulo = f"$\\rho$ = {r:g} (polo = {polo:.3f})"
        if fator == 1.0:
            rotulo += " — escolhido"
        plt.plot(f, 20 * np.log10(np.maximum(mag, 1e-12)), estilo, label=rotulo)
    plt.xlabel("Frequência (ciclos/semana)"); plt.ylabel("Magnitude (dB)")
    plt.title(f"Controlador H=1 como filtro passa-baixa ($\\lambda$ = {lamb:g})")
    plt.grid(True); plt.legend()
    _salvar_figura(output_dir, "resposta_frequencia_controlador.png")


def grafico_exposicao_vol(res_vol, sigma, datas, sigma_alvo, output_dir):
    fig, eixo_u = plt.subplots(figsize=(12, 5))
    eixo_u.plot(res_vol.index, res_vol["posicao_u"], color="C0", lw=1.6)
    eixo_u.set_xlabel("Data"); eixo_u.set_ylabel("Exposição u", color="C0")
    eixo_u.set_ylim(-0.05, 1.05)
    eixo_vol = eixo_u.twinx()
    eixo_vol.plot(datas, 100 * sigma, color="C3", lw=1.4, linestyle="--")
    eixo_vol.axhline(100 * sigma_alvo, color="gray", lw=0.8, linestyle=":")
    eixo_vol.set_ylabel("Vol prevista (% a.a.)", color="C3")
    plt.title("Exposição recua quando a volatilidade prevista sobe")
    _salvar_figura(output_dir, "exposicao_vs_volatilidade.png")
