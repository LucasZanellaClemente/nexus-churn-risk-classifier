"""
Testa se aumentar o dataset de sarcasmo (com padrão estrutural reconhecível)
melhora a detecção, e demonstra o VotingClassifier combinando NB + LR.
"""
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

marcadores = ["Que ótimo", "Adorei", "Sensacional", "Maravilhoso", "Perfeito",
              "Que alívio", "Nossa, que sorte", "Fantástico", "Show", "Excelente"]

eventos_negativos = [
    "o sistema caiu bem na hora da reunião", "a atualização quebrou tudo",
    "o suporte não respondeu por duas semanas", "perdemos os dados importantes",
    "o chamado ficou sem resposta por um mês", "o módulo travou de novo",
    "a licença ficou mais cara e não recebemos nada a mais",
    "o backup falhou quando mais precisávamos",
    "o consultor sumiu no meio da implantação", "o relatório saiu errado outra vez",
]
eventos_positivos = [
    "o sistema funcionou perfeitamente esse mês", "o suporte resolveu nosso problema rapidamente",
    "a nova versão trouxe recursos úteis", "a equipe conseguiu se adaptar bem ao sistema",
    "o treinamento ajudou bastante a equipe", "a implantação foi concluída no prazo",
    "o relatório saiu correto dessa vez", "o backup funcionou quando precisamos",
    "o consultor foi muito atencioso", "a licença manteve o mesmo valor",
]

# classe 1: marcador positivo + evento NEGATIVO = sarcasmo
sarcastico = [f"{m}, {e}." for m, e in itertools.product(marcadores, eventos_negativos)]

# classe 0: marcador positivo + evento POSITIVO (elogio sincero)
positivo_sincero = [f"{m}, {e}." for m, e in itertools.product(marcadores, eventos_positivos)]
# classe 0 também: eventos ditos diretamente, sem marcador (literal simples)
literal_direto = [e[0].upper() + e[1:] + "." for e in eventos_negativos + eventos_positivos]

df = pd.DataFrame({
    "frase": sarcastico + positivo_sincero + literal_direto,
    "sarcasmo": [1] * len(sarcastico) + [0] * (len(positivo_sincero) + len(literal_direto)),
})
df = df.drop_duplicates(subset="frase").sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset ampliado: {len(df)} frases | distribuição:\n{df['sarcasmo'].value_counts()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    df["frase"], df["sarcasmo"], test_size=0.2, random_state=42, stratify=df["sarcasmo"]
)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
voting = VotingClassifier(estimators=[("nb", nb), ("lr", lr)], voting="soft")

for nome, modelo in [("Naive Bayes", nb), ("Logistic Regression", lr), ("Voting (NB+LR)", voting)]:
    modelo.fit(X_train_vec, y_train)
    pred = modelo.predict(X_test_vec)
    print(f"--- {nome} ---")
    print(f"Acurácia: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=["Literal", "Sarcástico"], zero_division=0))
