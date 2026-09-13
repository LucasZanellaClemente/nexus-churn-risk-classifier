"""
Passo 5: Análise dos erros.
Foco nos dois modelos da comparação oficial: Naive Bayes e Logistic Regression.
- Lista os falsos positivos e falsos negativos (frases reais).
- Mostra as palavras mais importantes pra cada modelo (feature importance).
"""
import numpy as np
import pandas as pd
import joblib

dados = joblib.load("../resultados/dados_passo3.joblib")
vectorizer = dados["vectorizer"]
modelos = dados["modelos"]
frases_teste = dados["X_teste_frases"].reset_index(drop=True)
y_teste = dados["y_teste"]
X_teste = vectorizer.transform(frases_teste)
vocab = vectorizer.get_feature_names_out()

print("=" * 70)
print("1) FALSOS POSITIVOS E FALSOS NEGATIVOS")
print("=" * 70)

for nome in ["Naive Bayes", "Logistic Regression"]:
    modelo = modelos[nome]
    X_in = np.clip(X_teste.toarray(), 0, None) if nome == "Naive Bayes" else X_teste
    pred = modelo.predict(X_in)

    fp_idx = np.where((pred == 1) & (y_teste == 0))[0]
    fn_idx = np.where((pred == 0) & (y_teste == 1))[0]

    print(f"\n--- {nome} ---")
    print(f"Falsos positivos (previu risco, mas não tinha): {len(fp_idx)}")
    for i in fp_idx[:5]:
        print(f"   • \"{frases_teste[i]}\"")

    print(f"Falsos negativos (não previu risco, mas tinha): {len(fn_idx)}")
    for i in fn_idx[:5]:
        print(f"   • \"{frases_teste[i]}\"")

print("\n" + "=" * 70)
print("2) PALAVRAS MAIS IMPORTANTES PRA CADA MODELO")
print("=" * 70)

# --- Logistic Regression: coeficientes ---
lr = modelos["Logistic Regression"]
coefs = lr.coef_[0]
top_risco = np.argsort(coefs)[-10:][::-1]
top_sem_risco = np.argsort(coefs)[:10]

print("\nLogistic Regression — palavras que mais PUXAM pra risco de churn:")
for i in top_risco:
    print(f"   {vocab[i]:20s} coef = {coefs[i]:+.3f}")

print("\nLogistic Regression — palavras que mais PUXAM pra sem risco:")
for i in top_sem_risco:
    print(f"   {vocab[i]:20s} coef = {coefs[i]:+.3f}")

# --- Naive Bayes: diferença de log-probabilidade entre classes ---
nb = modelos["Naive Bayes"]
diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]  # >0 favorece classe 1 (risco)
top_risco_nb = np.argsort(diff)[-10:][::-1]
top_sem_risco_nb = np.argsort(diff)[:10]

print("\nNaive Bayes — palavras que mais indicam risco de churn:")
for i in top_risco_nb:
    print(f"   {vocab[i]:20s} diff = {diff[i]:+.3f}")

print("\nNaive Bayes — palavras que mais indicam ausência de risco:")
for i in top_sem_risco_nb:
    print(f"   {vocab[i]:20s} diff = {diff[i]:+.3f}")
