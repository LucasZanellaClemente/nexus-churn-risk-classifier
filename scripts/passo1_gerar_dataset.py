"""
Expande o dataset sintético de frases x risco de churn (0/1) pra um volume
bem maior, combinando frases-base (mais naturais) com geração por templates
(produto x sujeito x tempo), removendo duplicatas, e só então separando
treino/teste (SMOTE entra depois, só no treino).
"""
import itertools
import random

import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)

# ---------- frases-base (as 110 originais, mais naturais) ----------
sem_risco_base = [
    "O sistema Protheus tem atendido muito bem nossas necessidades no financeiro.",
    "Estamos satisfeitos com a implementação do módulo de estoque.",
    "A equipe de suporte resolveu nosso chamado rapidamente.",
    "Vamos renovar o contrato sem nenhuma ressalva.",
    "O relatório gerencial ficou exatamente como esperávamos.",
    "Gostaríamos de expandir o uso do RM para o setor de RH.",
    "A integração com o Fluig facilitou muito nosso fluxo de aprovações.",
    "Nossa equipe já domina bem o sistema depois do treinamento.",
    "O desempenho do sistema melhorou muito após a última atualização.",
    "Estamos animados com os novos recursos anunciados para o próximo release.",
    "O consultor foi muito atencioso durante a implantação.",
    "Conseguimos reduzir o tempo de fechamento contábil pela metade.",
    "A automação dos processos fiscais superou nossas expectativas.",
    "Queremos agendar um treinamento adicional para os novos colaboradores.",
    "O painel de indicadores facilitou muito a tomada de decisão.",
    "Estamos avaliando contratar mais licenças para outra filial.",
    "O suporte técnico costuma responder em poucas horas.",
    "A migração de dados foi feita sem nenhum problema.",
    "Ficamos impressionados com a facilidade de uso do módulo de vendas.",
    "Nossa diretoria está satisfeita com os resultados até agora.",
    "O sistema tem se mostrado estável mesmo em períodos de pico.",
    "Gostaríamos de conhecer mais sobre o módulo de BI.",
    "A parceria com a TOTVS tem sido muito positiva.",
    "Recomendamos a solução para outras empresas do grupo.",
    "O time de implantação cumpriu todos os prazos combinados.",
    "Estamos confiantes de que o sistema vai atender nosso crescimento.",
    "A personalização atendeu exatamente o que pedimos.",
    "Não tivemos nenhum problema desde a última atualização.",
    "O custo-benefício da solução continua sendo muito bom para nós.",
    "Vamos manter o mesmo plano para o próximo ano.",
    "A equipe financeira elogiou o novo módulo de contas a pagar.",
    "Estamos satisfeitos com o nível de suporte oferecido.",
    "O sistema se integrou bem com nossas outras ferramentas.",
    "Já indicamos a TOTVS para outros parceiros do setor.",
    "Conseguimos automatizar boa parte dos processos manuais.",
    "A experiência com o Fluig tem sido muito positiva.",
    "O novo dashboard facilitou o acompanhamento das metas.",
    "Não há planos de trocar de fornecedor no momento.",
    "A implantação do módulo fiscal foi tranquila.",
    "Estamos satisfeitos com a evolução do produto ao longo dos anos.",
    "O gerente de contas tem sido muito proativo conosco.",
    "Queremos ampliar o uso do sistema para mais departamentos.",
    "A curva de aprendizado foi mais rápida do que esperávamos.",
    "O suporte remoto resolveu nosso problema no mesmo dia.",
    "Continuamos confiantes na parceria de longo prazo.",
    "A atualização trouxe funcionalidades que estávamos pedindo há tempos.",
    "Nosso time está satisfeito com a usabilidade da plataforma.",
    "Vamos assinar o novo módulo de gestão de projetos.",
    "O processo de renovação foi simples e rápido.",
    "Estamos confortáveis com o nível de investimento atual.",
    "A qualidade do atendimento superou nossas expectativas.",
    "Conseguimos cumprir o fechamento fiscal sem atrasos.",
    "O treinamento da equipe foi muito bem avaliado internamente.",
    "Temos planos de expandir o contrato para outra unidade.",
    "O suporte prioritário tem funcionado muito bem para nós.",
    "Estamos satisfeitos com o retorno sobre o investimento até aqui.",
    "A nova versão do Protheus trouxe ganhos reais de produtividade.",
    "Nosso time de TI elogiou a estabilidade do ambiente.",
    "Vamos manter o mesmo nível de suporte contratado.",
    "A experiência geral com a TOTVS tem sido positiva.",
    "Queremos entender melhor os recursos de inteligência artificial do sistema.",
    "O processo de onboarding foi bem estruturado.",
    "Conseguimos reduzir erros manuais depois da automação.",
    "A equipe está confiante para expandir o uso do sistema.",
    "Não identificamos nenhum ponto crítico até o momento.",
    "O suporte técnico esclareceu todas as nossas dúvidas.",
    "Estamos satisfeitos com o cumprimento do SLA.",
    "A colaboração entre as equipes tem funcionado bem.",
    "Vamos revisar o contrato apenas para adicionar novos usuários.",
    "O sistema atendeu bem ao pico de vendas do fim de ano.",
]

com_risco_base = [
    "Estamos avaliando outras opções no mercado.",
    "O suporte tem demorado demais para responder nossos chamados.",
    "O custo da licença aumentou muito e estamos reconsiderando.",
    "Tivemos vários problemas de instabilidade no sistema esse mês.",
    "Um concorrente nos apresentou uma proposta mais em conta.",
    "A equipe está insatisfeita com a complexidade do módulo fiscal.",
    "Não recebemos retorno sobre o chamado aberto há duas semanas.",
    "Estamos pensando em migrar para outra plataforma.",
    "O treinamento não foi suficiente e a equipe está com dificuldades.",
    "A diretoria está questionando se vale a pena continuar com o contrato.",
    "O sistema tem apresentado lentidão constante nos últimos dias.",
    "Estamos recebendo muitas reclamações internas sobre o suporte.",
    "O valor da renovação ficou acima do que esperávamos pagar.",
    "Perdemos dados importantes em uma das últimas atualizações.",
    "A implementação está atrasada há mais de dois meses.",
    "Estamos considerando cancelar um dos módulos contratados.",
    "O consultor não tem conseguido resolver nosso problema.",
    "A equipe está frustrada com a curva de aprendizado do sistema.",
    "Recebemos uma proposta melhor de outro fornecedor esta semana.",
    "O chamado de suporte já está aberto há mais de dez dias sem solução.",
    "Não estamos satisfeitos com o nível de personalização oferecido.",
    "A diretoria pediu uma análise de custo-benefício antes de renovar.",
    "O sistema caiu duas vezes essa semana durante o horário comercial.",
    "Estamos revendo o orçamento e o contrato pode ser um dos cortes.",
    "A comunicação com o time de suporte tem sido falha.",
    "Precisamos de uma solução mais barata para o próximo ciclo.",
    "O módulo de vendas não atende mais o volume que temos hoje.",
    "Estamos insatisfeitos com o tempo de resposta do suporte técnico.",
    "A concorrência tem oferecido planos mais flexíveis.",
    "Tivemos que recontratar a implantação porque não funcionou como prometido.",
    "A equipe reclama que o sistema é lento durante o fechamento mensal.",
    "Estamos avaliando se vale a pena continuar com todos os módulos.",
    "O relacionamento com o time de contas piorou nos últimos meses.",
    "Recebemos reclamações recorrentes sobre erros no módulo fiscal.",
    "A diretoria está insatisfeita com o retorno sobre o investimento.",
    "Estamos sem suporte adequado desde a saída do nosso consultor dedicado.",
    "O contrato está para vencer e ainda não decidimos se renovamos.",
    "Tivemos problemas recorrentes de integração com outros sistemas.",
    "A equipe financeira quer reduzir custos com softwares no próximo ano.",
    "Estamos insatisfeitos com a falta de atualizações prometidas.",
]

# ---------- geração por templates (produto x sujeito x tempo) ----------
produtos = [
    "o sistema Protheus", "o módulo de RM", "a plataforma Fluig", "o ERP",
    "a plataforma", "o sistema", "o módulo financeiro", "o módulo fiscal",
    "o módulo de vendas", "o módulo de estoque", "a solução da TOTVS",
    "o sistema de gestão", "o módulo de RH", "o módulo de compras",
]
sujeitos = [
    "nossa equipe", "a diretoria", "o time financeiro", "nosso time de TI",
    "a equipe comercial", "o setor de RH", "a equipe de operações",
    "nossa gerência", "o time de suporte interno", "a equipe de projetos",
    "o time de logística", "a equipe de compras",
]
tempos = [
    "essa semana", "esse mês", "nos últimos dias", "recentemente",
    "há duas semanas", "no último trimestre", "desde a última atualização",
    "nas últimas reuniões", "ao longo do ano", "nos últimos meses",
]

templates_sem_risco = [
    "{sujeito} está muito satisfeita com {produto} {tempo}.",
    "{produto} tem atendido bem nossas necessidades {tempo}.",
    "{sujeito} elogiou o desempenho de {produto} {tempo}.",
    "Não tivemos nenhum problema com {produto} {tempo}.",
    "{sujeito} está confiante na parceria com a TOTVS.",
    "{sujeito} quer expandir o uso de {produto} para outros setores.",
    "O suporte resolveu rapidamente nosso chamado sobre {produto}.",
    "{sujeito} recomendaria {produto} para outras empresas.",
    "A experiência com {produto} tem sido muito positiva {tempo}.",
    "{sujeito} está animada com os novos recursos de {produto}.",
    "{sujeito} conseguiu automatizar processos importantes com {produto}.",
    "{produto} se mostrou estável mesmo em períodos de pico {tempo}.",
]

templates_com_risco = [
    "{sujeito} está avaliando trocar {produto} por outra solução.",
    "{produto} apresentou instabilidade {tempo}.",
    "{sujeito} reclamou da demora do suporte sobre {produto} {tempo}.",
    "O custo de {produto} aumentou e {sujeito} está reconsiderando o contrato.",
    "Um concorrente ofereceu uma proposta melhor que {produto}.",
    "{sujeito} está insatisfeita com {produto} {tempo}.",
    "Não recebemos retorno do chamado sobre {produto} {tempo}.",
    "{sujeito} está pensando em migrar para outra plataforma.",
    "{produto} apresentou lentidão constante {tempo}.",
    "{sujeito} questiona se vale a pena continuar pagando por {produto}.",
    "Tivemos perda de dados usando {produto} {tempo}.",
    "{sujeito} está frustrada com a complexidade de {produto}.",
]


def gerar(templates):
    # usa dict em vez de set: um set() itera em ordem de hash, que muda a
    # cada execução do Python (PYTHONHASHSEED aleatório) e quebrava a
    # reprodutibilidade. dict.fromkeys preserva a ordem de inserção.
    vistas = {}
    combos = list(itertools.product(templates, produtos, sujeitos, tempos))
    random.shuffle(combos)
    for template, produto, sujeito, tempo in combos:
        frase = template.format(produto=produto, sujeito=sujeito, tempo=tempo)
        frase = frase[0].upper() + frase[1:]
        vistas[frase] = None
    return list(vistas.keys())


sem_risco_gerado = gerar(templates_sem_risco)
com_risco_gerado = gerar(templates_com_risco)

# ---------- alvo de volume: mantém proporção original (~64% / ~36%) ----------
TOTAL = 3000
N_SEM_RISCO = int(TOTAL * 0.64)
N_COM_RISCO = TOTAL - N_SEM_RISCO

sem_risco_final = list(dict.fromkeys(sem_risco_base + sem_risco_gerado))[:N_SEM_RISCO]
com_risco_final = list(dict.fromkeys(com_risco_base + com_risco_gerado))[:N_COM_RISCO]

df = pd.DataFrame(
    {
        "frase": sem_risco_final + com_risco_final,
        "risco_churn": [0] * len(sem_risco_final) + [1] * len(com_risco_final),
    }
)
df = df.drop_duplicates(subset="frase").sample(frac=1, random_state=42).reset_index(drop=True)

treino, teste = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["risco_churn"]
)

treino.to_csv("../dados/transcricoes_treino.csv", index=False, encoding="utf-8")
teste.to_csv("../dados/transcricoes_teste.csv", index=False, encoding="utf-8")

print("Total:", df.shape, "| distribuição geral:\n", df["risco_churn"].value_counts())
print("\nTreino:", treino.shape, "| distribuição:\n", treino["risco_churn"].value_counts())
print("\nTeste:", teste.shape, "| distribuição:\n", teste["risco_churn"].value_counts())
