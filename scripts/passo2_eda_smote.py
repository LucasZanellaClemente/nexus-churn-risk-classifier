"""
Passo 2: Análise exploratória do CSV + balanceamento com SMOTE (só no treino).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def smote_manual(X_min, n_samples, k=5, random_state=42):
    """Implementação simples do SMOTE: interpola entre vizinhos mais
    próximos da classe minoritária. Usada aqui porque a biblioteca
    imbalanced-learn não pôde ser instalada neste ambiente (sem acesso
    ao pypi no momento) — o algoritmo é o mesmo, só sem o pacote pronto."""
    rng = np.random.RandomState(random_state)
    n_min = X_min.shape[0]
    k_eff = min(k, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    _, indices = nn.kneighbors(X_min)
    synthetic = np.empty((n_samples, X_min.shape[1]))
    for s in range(n_samples):
        i = rng.randint(0, n_min)
        viz = indices[i, rng.randint(1, k_eff + 1)]  # pula o próprio ponto (posição 0)
        lam = rng.rand()
        synthetic[s] = X_min[i] + lam * (X_min[viz] - X_min[i])
    return synthetic

treino = pd.read_csv("../dados/transcricoes_treino.csv")
teste = pd.read_csv("../dados/transcricoes_teste.csv")

print("=" * 60)
print("1) CONTAGEM E DISTRIBUIÇÃO DAS CLASSES (treino)")
print("=" * 60)
print(treino["risco_churn"].value_counts())
print(treino["risco_churn"].value_counts(normalize=True).round(3) * 100, "%")

print("\n" + "=" * 60)
print("2) TAMANHO DAS FRASES POR CLASSE (nº de palavras)")
print("=" * 60)
treino["tamanho"] = treino["frase"].str.split().str.len()
print(treino.groupby("risco_churn")["tamanho"].describe()[["mean", "min", "max"]])

print("\n" + "=" * 60)
print("3) PALAVRAS MAIS FREQUENTES POR CLASSE")
print("=" * 60)
stop_pt = ["o", "a", "os", "as", "de", "da", "do", "com", "que", "para",
           "está", "estão", "e", "no", "na", "nos", "nas", "um", "uma",
           "não", "tem", "temos", "foi", "ser", "mais", "muito"]
for classe, nome in [(0, "SEM risco"), (1, "COM risco")]:
    textos = treino[treino["risco_churn"] == classe]["frase"]
    cv = CountVectorizer(stop_words=stop_pt, max_features=10)
    cv.fit(textos)
    print(f"{nome}: {list(cv.get_feature_names_out())}")

print("\n" + "=" * 60)
print("4) DUPLICATAS E INCONSISTÊNCIAS")
print("=" * 60)
print("Frases duplicadas:", treino["frase"].duplicated().sum())
conflitos = treino.groupby("frase")["risco_churn"].nunique().gt(1).sum()
print("Frases com rótulo conflitante:", conflitos)

# ---------- gráficos ----------
fig, ax = plt.subplots(figsize=(4, 3.2))
treino["risco_churn"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
)
ax.set_xticklabels(["Sem risco (0)", "Com risco (1)"], rotation=0)
ax.set_ylabel("Quantidade de frases")
ax.set_title("Distribuição das classes (treino, antes do SMOTE)")
fig.tight_layout()
fig.savefig("../resultados/distribuicao_classes.png", dpi=150)

fig2, ax2 = plt.subplots(figsize=(4, 3.2))
treino[treino.risco_churn == 0]["tamanho"].plot(kind="hist", alpha=0.6, label="Sem risco", ax=ax2, bins=10)
treino[treino.risco_churn == 1]["tamanho"].plot(kind="hist", alpha=0.6, label="Com risco", ax=ax2, bins=10)
ax2.set_xlabel("Nº de palavras na frase")
ax2.set_title("Tamanho das frases por classe")
ax2.legend()
fig2.tight_layout()
fig2.savefig("../resultados/tamanho_frases.png", dpi=150)

# ---------- vetorização + SMOTE (só no treino) ----------
print("\n" + "=" * 60)
print("5) TF-IDF + SMOTE (aplicado só no treino)")
print("=" * 60)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(treino["frase"]).toarray()
y_train = treino["risco_churn"].to_numpy()

contagem = pd.Series(y_train).value_counts()
print("Antes do SMOTE:", dict(contagem))

classe_maioria = contagem.idxmax()
classe_minoria = contagem.idxmin()
n_faltando = contagem[classe_maioria] - contagem[classe_minoria]

X_minoria = X_train[y_train == classe_minoria]
sinteticos = smote_manual(X_minoria, n_samples=n_faltando, k=5, random_state=42)

X_train_bal = np.vstack([X_train, sinteticos])
y_train_bal = np.concatenate([y_train, np.full(n_faltando, classe_minoria)])

print("Depois do SMOTE:", dict(pd.Series(y_train_bal).value_counts()))
print("Formato da matriz de treino balanceada:", X_train_bal.shape)
print("\nTeste continua intocado:", dict(teste["risco_churn"].value_counts()))

# salva pra reaproveitar no Passo 3 (modelagem)
import joblib
joblib.dump(
    {
        "vectorizer": vectorizer,
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "X_teste_frases": teste["frase"],
        "y_teste": teste["risco_churn"].to_numpy(),
    },
    "../resultados/dados_passo2.joblib",
)
