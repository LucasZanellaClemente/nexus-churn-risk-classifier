"""
Passo 5 (complemento) — Experimento exploratório: detecção de sarcasmo/ironia.
IMPORTANTE: este é um experimento à parte do modelo principal de churn,
com escopo e limitações reconhecidas (ver comentários). Não deve ser
apresentado como um recurso robusto do sistema, e sim como uma investigação
que ajuda a entender os limites da abordagem TF-IDF + ML clássico.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ---------- dataset pequeno e claramente sintético ----------
# sarcástico (1): usa palavras positivas pra dizer algo negativo
sarcastico = [
    "Ah, que ótimo, mais uma atualização quebrada bem na hora do fechamento.",
    "Adoramos esperar duas semanas por uma resposta do suporte, sério.",
    "Perfeito, o sistema caiu de novo bem no meio da reunião importante.",
    "Nossa, que suporte excelente, ainda esperando desde a semana passada.",
    "Que maravilha, mais uma licença mais cara pra pagar por menos recursos.",
    "Claro, porque perder os dados foi exatamente o que precisávamos hoje.",
    "Show, o treinamento resolveu tudo, ninguém sabe usar o sistema ainda.",
    "Ótimo trabalho, o relatório saiu errado igual da última vez.",
    "Excelente, o chamado que abrimos há um mês continua sem resposta.",
    "Que sorte a nossa, o módulo trava bem quando mais precisamos dele.",
    "Uau, adorei ter que reexplicar o mesmo problema pela quinta vez.",
    "Fico muito feliz em pagar mais caro por um suporte que nunca responde.",
    "Que ótima surpresa, o sistema lento bem no fechamento do mês.",
    "Maravilhoso, perdemos a reunião toda por causa da instabilidade.",
    "Sensacional, o consultor sumiu bem no meio da implantação.",
    "Adorei a nova versão, quebrou tudo que já funcionava antes.",
    "Que alívio, mais um bug pra descobrir sozinhos sem ajuda do suporte.",
    "Fantástico, pagamos caro por um recurso que nem funciona direito.",
    "Que ótimo, o backup falhou justo quando precisamos dele.",
    "Adoro quando o sistema trava sem nenhum aviso, muito prático.",
]

# literal (0): frases diretas, positivas ou negativas, sem ironia
literal = [
    "O suporte demorou duas semanas pra responder nosso chamado.",
    "Estamos satisfeitos com o desempenho do sistema esse mês.",
    "O sistema caiu durante a reunião importante.",
    "A licença ficou mais cara nessa renovação.",
    "Perdemos dados importantes na última atualização.",
    "O treinamento não foi suficiente pra equipe usar o sistema bem.",
    "O relatório saiu com erro novamente.",
    "O chamado aberto há um mês ainda não teve resposta.",
    "Gostamos muito da nova versão do sistema.",
    "O consultor foi muito atencioso durante a implantação.",
    "Conseguimos resolver o problema rapidamente com o suporte.",
    "A equipe está satisfeita com os novos recursos.",
    "O backup funcionou perfeitamente quando precisamos.",
    "O módulo de vendas está funcionando bem.",
    "Estamos pensando em migrar para outra plataforma.",
    "O sistema apresentou instabilidade essa semana.",
    "A implantação foi concluída dentro do prazo.",
    "Recebemos um bom atendimento do time de suporte.",
    "O custo da licença está dentro do esperado.",
    "A equipe elogiou a facilidade de uso do sistema.",
]

df = pd.DataFrame(
    {"frase": sarcastico + literal, "sarcasmo": [1] * len(sarcastico) + [0] * len(literal)}
)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["frase"], df["sarcasmo"], test_size=0.3, random_state=42, stratify=df["sarcasmo"]
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Dataset de sarcasmo: {len(df)} frases ({len(sarcastico)} sarcásticas / {len(literal)} literais)")
print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}\n")

for nome, modelo in [
    ("Naive Bayes", MultinomialNB()),
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
]:
    modelo.fit(X_train_vec, y_train)
    pred = modelo.predict(X_test_vec)
    print(f"--- {nome} (detecção de sarcasmo) ---")
    print(classification_report(y_test, pred, target_names=["Literal", "Sarcástico"], zero_division=0))
    print("Matriz de confusão:\n", confusion_matrix(y_test, pred))
    print()
