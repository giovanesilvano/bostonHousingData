# Main.py
# Análise estatística do dataset Boston Housing (HousingData.csv)
# Fonte: U.S. Census Service / Kaggle - The Boston Housing Dataset
#
# Para cada coluna numérica: média, mediana, moda, desvio padrão,
# coeficiente de variação, quartis e teste de normalidade
# Kolmogorov-Smirnov, com um histograma (PNG) claro e descritivo.
# Para colunas categóricas (CHAS): moda e frequências.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # gera imagens sem abrir janelas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import kstest, norm

# ------------------------------------------------------------------
# 0. Dicionário de dados (descrição de cada coluna, em português)
# ------------------------------------------------------------------
DESCRICAO = {
    "CRIM":    ("Taxa de criminalidade per capita por cidade", "crimes per capita"),
    "ZN":      ("Proporção de terrenos residenciais com lotes > 25.000 pés²", "%"),
    "INDUS":   ("Proporção de acres de negócios não varejistas por cidade", "%"),
    "CHAS":    ("Margeia o rio Charles (1 = sim, 0 = não)", ""),
    "NOX":     ("Concentração de óxidos nítricos", "partes por 10 milhões"),
    "RM":      ("Número médio de cômodos por residência", "cômodos"),
    "AGE":     ("Proporção de imóveis ocupados pelo dono construídos antes de 1940", "%"),
    "DIS":     ("Distância ponderada a 5 centros de emprego de Boston", "distância ponderada"),
    "RAD":     ("Índice de acessibilidade a rodovias radiais", "índice"),
    "TAX":     ("Taxa de imposto predial por US$ 10.000", "US$ por 10.000"),
    "PTRATIO": ("Razão alunos/professor por cidade", "alunos por professor"),
    "B":       ("1000·(Bk − 0,63)², Bk = proporção de população negra por cidade", "índice"),
    "LSTAT":   ("% da população de baixa renda", "%"),
    "MEDV":    ("Valor mediano dos imóveis ocupados pelo dono", "US$ mil"),
}

ROTULO_CHAS = {0: "Não margeia o rio", 1: "Margeia o rio Charles"}

# ------------------------------------------------------------------
# 1. Carregamento do arquivo (mesma pasta do Main.py)
# ------------------------------------------------------------------
PASTA = Path(__file__).resolve().parent
CAMINHO_CSV = PASTA / "HousingData.csv"
PASTA_GRAFICOS = PASTA / "histogramas"
PASTA_GRAFICOS.mkdir(exist_ok=True)

NIVEL_SIGNIFICANCIA = 0.05

df = pd.read_csv(CAMINHO_CSV)

print("=" * 72)
print("VISÃO GERAL DO DATASET - BOSTON HOUSING")
print("=" * 72)
print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}\n")
print(df.head(), "\n")
print("Dicionário de dados:")
for col in df.columns:
    desc, uni = DESCRICAO.get(col, ("(sem descrição)", ""))
    print(f"  {col:<8} {desc}" + (f" [{uni}]" if uni else ""))
print("\nValores ausentes por coluna:")
print(df.isna().sum(), "\n")

# ------------------------------------------------------------------
# 2. Separação numéricas x categóricas
# ------------------------------------------------------------------
# CHAS é uma dummy (0/1) -> tratada como categórica.
for col in ["CHAS"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
colunas_categoricas = df.select_dtypes(
    include=["object", "category", "bool"]).columns.tolist()

print(f"Colunas numéricas   : {colunas_numericas}")
print(f"Colunas categóricas : {colunas_categoricas}\n")

# ------------------------------------------------------------------
# 3. Análise das colunas numéricas
# ------------------------------------------------------------------
resultados = []

for coluna in colunas_numericas:
    dados = df[coluna].dropna()
    if dados.empty:
        continue

    descricao, unidade = DESCRICAO.get(coluna, (coluna, ""))
    n = int(dados.count())

    # Concentração
    media = dados.mean()
    mediana = dados.median()
    modas = dados.mode().tolist()
    moda = modas[0]
    moda_str = ", ".join(f"{m:.4g}" for m in modas[:3])
    if len(modas) > 3:
        moda_str += f" (+{len(modas) - 3})"

    # Dispersão
    desvio_padrao = dados.std(ddof=1)
    variancia = dados.var(ddof=1)
    coef_variacao = (desvio_padrao / abs(media)) * 100 if media != 0 else np.nan
    if coef_variacao < 15:
        dispersao = "baixa (homogêneo)"
    elif coef_variacao < 30:
        dispersao = "moderada"
    else:
        dispersao = "alta (heterogêneo)"

    pct_1dp = ((dados >= media - desvio_padrao) & (dados <= media + desvio_padrao)).mean() * 100
    pct_2dp = ((dados >= media - 2 * desvio_padrao) & (dados <= media + 2 * desvio_padrao)).mean() * 100

    # Quartis
    minimo, maximo = dados.min(), dados.max()
    q1, q2, q3 = dados.quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = dados[(dados < lim_inf) | (dados > lim_sup)]
    pct_out = 100 * len(outliers) / n

    # Forma
    assimetria = dados.skew()
    curtose = dados.kurtosis()
    if abs(assimetria) < 0.5:
        forma = "aproximadamente simétrica"
    elif assimetria > 0:
        forma = "assimétrica à direita (cauda longa para valores altos)"
    else:
        forma = "assimétrica à esquerda (cauda longa para valores baixos)"

    # Teste Kolmogorov-Smirnov (H0: normal)
    if desvio_padrao > 0:
        z = (dados - media) / desvio_padrao
        ks_stat, ks_p = kstest(z, "norm")
        eh_normal = ks_p >= NIVEL_SIGNIFICANCIA
    else:
        ks_stat, ks_p, eh_normal = np.nan, np.nan, False
    normal = ("Não rejeita H0 → compatível com a normal" if eh_normal
              else "Rejeita H0 → distribuição NÃO normal")

    resultados.append({
        "coluna": coluna, "descricao": descricao, "n": n,
        "media": media, "mediana": mediana, "moda": moda_str,
        "desvio_padrao": desvio_padrao, "variancia": variancia,
        "coef_variacao_%": coef_variacao,
        "minimo": minimo, "Q1": q1, "Q2": q2, "Q3": q3, "maximo": maximo,
        "IQR": iqr, "outliers": len(outliers),
        "assimetria": assimetria, "curtose": curtose,
        "KS_D": ks_stat, "KS_p_valor": ks_p, "normalidade": normal,
    })

    # --- Impressão ---
    print("-" * 72)
    print(f"{coluna} — {descricao}" + (f" [{unidade}]" if unidade else ""))
    print(f"n = {n}")
    print("-" * 72)
    print("  CONCENTRAÇÃO")
    print(f"    Média = {media:.4f} | Mediana = {mediana:.4f} | Moda = {moda_str}")
    print("  DISPERSÃO")
    print(f"    Desvio padrão = {desvio_padrao:.4f} | Variância = {variancia:.4f}")
    print(f"    Coef. de variação = {coef_variacao:.2f}% → dispersão {dispersao}")
    print(f"    Dentro de média ± 1 DP: {pct_1dp:.1f}% | ± 2 DP: {pct_2dp:.1f}% "
          f"(normal ≈ 68% / 95%)")
    print("  QUARTIS")
    print(f"    Mín = {minimo:.4f} | Q1 = {q1:.4f} | Q2 = {q2:.4f} | "
          f"Q3 = {q3:.4f} | Máx = {maximo:.4f}")
    print(f"    IQR = {iqr:.4f} | outliers (1,5·IQR) = {len(outliers)} ({pct_out:.1f}%)")
    print(f"    Forma: {forma} (assimetria = {assimetria:.2f}, curtose = {curtose:.2f})")
    print("  NORMALIDADE — KOLMOGOROV-SMIRNOV (α = 0,05)")
    print(f"    D = {ks_stat:.4f} | p-valor = {ks_p:.6f} → {normal}\n")

    # ------------------------------------------------------------------
    # HISTOGRAMA
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(right=0.70)          # espaço à direita p/ o painel

    # Faixas de fundo
    ax.axvspan(q1, q3, color="#8e6bbf", alpha=0.13, zorder=0)                     # Q1–Q3
    ax.axvspan(media - desvio_padrao, media + desvio_padrao,
               color="#e63946", alpha=0.07, zorder=0)                              # média ± DP

    # Histograma
    ax.hist(dados, bins="auto", density=True, color="#a8c8e8",
            edgecolor="#2b4c7e", linewidth=0.8, alpha=0.9, zorder=2)

    # Curva normal teórica
    if desvio_padrao > 0:
        x = np.linspace(minimo, maximo, 400)
        ax.plot(x, norm.pdf(x, media, desvio_padrao), color="black",
                linewidth=2.2, zorder=4)

    # Linhas verticais
    ax.axvline(q1, color="#6a3fa0", linestyle="-.", linewidth=1.8, zorder=3)
    ax.axvline(q3, color="#6a3fa0", linestyle="-.", linewidth=1.8, zorder=3)
    ax.axvline(mediana, color="#2a9d3f", linestyle="--", linewidth=2.4, zorder=5)
    ax.axvline(media, color="#e63946", linestyle="--", linewidth=2.4, zorder=5)
    ax.axvline(moda, color="#f4a261", linestyle=":", linewidth=2.4, zorder=5)

    # Rótulos das linhas, em alturas diferentes para não se sobrepor
    ymax = ax.get_ylim()[1]
    rotulos = [
        (q1, f"Q1\n{q1:.2f}", "#6a3fa0", 0.99),
        (q3, f"Q3\n{q3:.2f}", "#6a3fa0", 0.99),
        (mediana, f"Mediana\n{mediana:.2f}", "#2a9d3f", 0.88),
        (media, f"Média\n{media:.2f}", "#e63946", 0.77),
        (moda, f"Moda\n{moda:.2f}", "#b5651d", 0.66),
    ]
    for valor, txt, cor, alt in rotulos:
        ax.annotate(txt, xy=(valor, ymax * alt), fontsize=9, color=cor,
                    fontweight="bold", ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=cor, alpha=0.9), zorder=6)

    # Legenda
    legenda = [
        Patch(facecolor="#a8c8e8", edgecolor="#2b4c7e", label="Histograma (densidade)"),
        Line2D([0], [0], color="black", lw=2.2, label="Curva normal teórica"),
        Line2D([0], [0], color="#e63946", ls="--", lw=2.4, label=f"Média = {media:.2f}"),
        Line2D([0], [0], color="#2a9d3f", ls="--", lw=2.4, label=f"Mediana (Q2) = {mediana:.2f}"),
        Line2D([0], [0], color="#f4a261", ls=":", lw=2.4, label=f"Moda = {moda_str}"),
        Line2D([0], [0], color="#6a3fa0", ls="-.", lw=1.8, label=f"Q1 = {q1:.2f}  |  Q3 = {q3:.2f}"),
        Patch(facecolor="#8e6bbf", alpha=0.25, label=f"Faixa Q1–Q3 (50% central, IQR = {iqr:.2f})"),
        Patch(facecolor="#e63946", alpha=0.15, label=f"Faixa média ± 1 DP (DP = {desvio_padrao:.2f})"),
    ]
    ax.legend(handles=legenda, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, frameon=True, title="Legenda", title_fontsize=10)

    # Painel de estatísticas + teste de normalidade
    cor_painel = "#d4edda" if eh_normal else "#f8d7da"
    veredito = "✔ DISTRIBUIÇÃO NORMAL" if eh_normal else "✘ DISTRIBUIÇÃO NÃO NORMAL"
    painel = (
        f"ESTATÍSTICAS (n = {n})\n"
        f"Média ........ {media:.3f}\n"
        f"Mediana ...... {mediana:.3f}\n"
        f"Moda ......... {moda_str}\n"
        f"Desvio padrão  {desvio_padrao:.3f}\n"
        f"Coef. var. ... {coef_variacao:.1f}%  ({dispersao})\n"
        f"Mín / Máx .... {minimo:.2f} / {maximo:.2f}\n"
        f"Q1 / Q2 / Q3 . {q1:.2f} / {q2:.2f} / {q3:.2f}\n"
        f"IQR .......... {iqr:.3f}\n"
        f"Outliers ..... {len(outliers)} ({pct_out:.1f}%)\n"
        f"Assimetria ... {assimetria:.2f}\n"
        f"Curtose ...... {curtose:.2f}\n"
        f"\nTESTE KOLMOGOROV-SMIRNOV\n"
        f"H0: dados ~ Normal\n"
        f"D = {ks_stat:.4f}\n"
        f"p-valor = {ks_p:.4f}  (α = 0,05)\n"
        f"{veredito}"
    )
    fig.text(0.715, 0.50, painel, fontsize=9, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=cor_painel,
                       edgecolor="gray", alpha=0.97))

    # Títulos e eixos
    fig.suptitle(f"{coluna} — {descricao}", fontsize=15, fontweight="bold",
                 x=0.36, y=0.985)
    ax.set_title(f"{veredito}  |  KS p = {ks_p:.4f}  |  {forma}",
                 fontsize=11, color=("#1e7e34" if eh_normal else "#b02a37"),
                 pad=10)
    ax.set_xlabel(f"{coluna}" + (f" ({unidade})" if unidade else ""), fontsize=12)
    ax.set_ylabel("Densidade de frequência", fontsize=12)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.set_ylim(0, ymax * 1.08)

    fig.savefig(PASTA_GRAFICOS / f"hist_{coluna}.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------
# 4. Tabela-resumo
# ------------------------------------------------------------------
tabela = pd.DataFrame(resultados)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

print("=" * 72)
print("RESUMO ESTATÍSTICO — COLUNAS NUMÉRICAS")
print("=" * 72)
print(tabela.drop(columns="descricao").to_string(index=False), "\n")

normais = tabela.loc[tabela["KS_p_valor"] >= NIVEL_SIGNIFICANCIA, "coluna"].tolist()
nao_normais = tabela.loc[tabela["KS_p_valor"] < NIVEL_SIGNIFICANCIA, "coluna"].tolist()
print("SÍNTESE DO TESTE DE NORMALIDADE (KS, α = 0,05)")
print(f"  Compatíveis com normal : {normais or 'nenhuma'}")
print(f"  Não normais            : {nao_normais}\n")

tabela.to_csv(PASTA / "resumo_estatistico.csv", index=False,
              sep=";", decimal=",", encoding="utf-8-sig")
print(f"Resumo salvo em    : {PASTA / 'resumo_estatistico.csv'}")
print(f"Gráficos salvos em : {PASTA_GRAFICOS}\n")

# ------------------------------------------------------------------
# 5. Colunas categóricas — moda e frequências
# ------------------------------------------------------------------
print("=" * 72)
print("COLUNAS CATEGÓRICAS — MODA E FREQUÊNCIAS")
print("=" * 72)

if not colunas_categoricas:
    print("Nenhuma coluna categórica encontrada.")

for coluna in colunas_categoricas:
    dados = df[coluna].dropna()
    descricao, _ = DESCRICAO.get(coluna, (coluna, ""))
    n = int(dados.count())
    modas = dados.mode().tolist()
    freq = dados.value_counts().sort_index()
    freq_rel = (freq / n * 100).round(2)

    def rotulo(v):
        try:
            return ROTULO_CHAS.get(int(v), str(v)) if coluna == "CHAS" else str(v)
        except (TypeError, ValueError):
            return str(v)

    moda_txt = ", ".join(f"{m} ({rotulo(m)})" for m in modas)

    print(f"\n{coluna} — {descricao}  (n = {n})")
    print(f"  Moda: {moda_txt}")
    print("  Frequências:")
    for cat in freq.index:
        print(f"    {cat} = {rotulo(cat):<22} {freq[cat]:>4}  ({freq_rel[cat]:.2f}%)")

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 5))
    cores = ["#e63946" if m in modas else "#a8c8e8" for m in freq.index]
    barras = ax.bar([f"{c}\n{rotulo(c)}" for c in freq.index], freq.values,
                    color=cores, edgecolor="#2b4c7e")
    for b, pct in zip(barras, freq_rel.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{int(b.get_height())}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"{coluna} — {descricao}\nModa = {moda_txt}  (barra destacada)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(coluna)
    ax.set_ylabel("Número de registros")
    ax.set_ylim(0, freq.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / f"barras_{coluna}.png", dpi=140)
    plt.close(fig)

print("\nAnálise concluída.")