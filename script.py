# -*- coding: utf-8 -*-
"""
===============================================================================
MBA USP ESALQ
Big Data e Deployment de Modelos I
===============================================================================
"""
# %%

# ========= Instruções para o funcionamento dos scripts das aulas =============
# =============================================================================
# No diretório que os arquivos foram carregados, será necessário criar um ambi-
# ente virtual digitando o seguinte comando no terminal
# =============================================================================
# Qualquer sistema operacional: python3 -m venv venv
# =============================================================================
# Logo em seguida, será necessário acessar o ambiente virtual criado digitando
# o seguinte comando no terminal
# =============================================================================
# Windows: .\venv\Scripts\activate
# Linux: /venv/bin/activate
# =============================================================================
# Após acessar o ambiente virtual, irá aparecer o nome (venv) entre parênteses
# na parte lateral esquerda do terminal
# Em seguida, será necessário instalar os pacotes nessessários digitando o se-
# guinte comando no terminal
# =============================================================================
# Qualquer sistema operacional: pip install -r requirements.txt
# =============================================================================
# Após a instalação de todos os pacotes, você já poderar inicializar o serviço
# do MLflow digitando o seguinte comando no terminal
# =============================================================================
# Qualquer sistema operacional: mlflow ui
# =============================================================================
# Com o MLflow inicializado, você poderá acessar a aplicação via browser digi-
# tando o seguinte endereço no seu browser de preferência
# =============================================================================
# No browser (Firefox, Chorme, outro de preferência): http://localhost:5000
# =============================================================================
# Caso localhost não funcione, tente:
# http://0.0.0.0:5000 ou http://127.0.0.1:5000
# =============================================================================

# %%

# Pacotes que serão utilizados em todo o script
import pandas as pd
import mlflow
import statsmodels.api as sm
import json
import requests
import matplotlib.pyplot as plt

# %%

# Coletando o caminho do dataset
caminho = "./datasets/tempodist.csv"

# Carregando o dataset
df_tempodist = pd.read_csv(caminho)

# %%

# Relembrando o dataset tempodist
df_tempodist

# %%

# Verificando as informações do dataset e possíveis valores nulos
df_tempodist.info()

# %%

# Verificando as variáveis descritivas das colunas
df_tempodist.describe()

# %% Regressão Linear 

# Treinando um modelo localmente
modelo_nulo_local = sm.OLS.from_formula(
    formula="tempo ~ distancia", data=df_tempodist).fit()

# %%

# Verificando seu output
modelo_nulo_local.summary()

# %%

# ==================== INICIALIZANDO COM O MLFLOW =============================

# Apontar a instância do mlflow para o servidor que está rodando a aplicação
mlflow.set_tracking_uri("http://localhost:8000")

# %%

# Criando o nosso primeiro experimento
mlflow.set_experiment(experiment_name="Regressão Linear Simples - tempodist")

# %% DATASET

# ================== DEFINIÇÃO DOS INPUTS DO MODELO ===========================

# Modelo proposto: Modelo nulo

# Coleta dos metadados do dataset para o padrão do mlflow
dataset = mlflow.data.from_pandas(df_tempodist)
# Fórmula que serão utilizada no modelo
formula = "tempo ~ 1"

# %%

# Criando o nosso primeiro experimento no mlflow
with mlflow.start_run(run_name="Modelo Nulo"):

    # ==================== INPUTS =============================================

    # Coleta dos metadados do dataset. O contexto pode ser utilizado para adi-
    # cionar mais uma informação de contexto do dataset, por exemplo: se ele é
    # um dataset de treino ou teste
    mlflow.log_input(dataset, context="training")

    # Parâmetros de entrada para serem coletados
    mlflow.log_param("Fórmula", formula)

    # ==================== MODELAGEM ==========================================

    # Treinamento do modelo
    modelo_nulo = sm.OLS.from_formula(formula=formula, data=df_tempodist).fit()

    # ==================== OUTPUTS ============================================

    # Registrando as métricas do modelo
    mlflow.log_metric("Estatística F", modelo_nulo.fvalue)
    mlflow.log_metric("F p-value", modelo_nulo.f_pvalue)
    mlflow.log_metric("R2", modelo_nulo.rsquared)

    # Coletando artefatos do modelo

    # Gráfico do comparativo dos dados observados versus fittedvalues
    fig, ax = plt.subplots()
    ax.scatter(df_tempodist["tempo"], modelo_nulo.fittedvalues)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", color="gray")

    # Registro do gráfico como um artefato do experimento
    mlflow.log_figure(
        fig, "observado_vs_fitted.png"
    )  # importante definir a extensão

    # Obtendo o sumário do modelo em formato de texto
    modelo_nulo_summary_texto = modelo_nulo.summary().as_text()

    # Registro do sumário como um artefato do experimento
    mlflow.log_text(
        modelo_nulo_summary_texto, "summary.txt"
    )  # importante definir a extensão do arquivo

    # Carregamento do modelo no experimento
    mlflow.statsmodels.log_model(modelo_nulo, "modelo-nulo")

# %%

# =================== UTILIZANDO MODELOS EM PRODUÇÃO ==========================

# Carregando o modelo tempo-distancia em estado staging (adaptação)
modelo_nulo_carregado = mlflow.statsmodels.load_model(
    "models:/tempo-distancia/staging"
)

# %%

# Verificando o R2
modelo_nulo_carregado.rsquared

# %%

# Verificando os fittedvalues
modelo_nulo_carregado.fittedvalues

# %%

# ============================ MODELO FINAL ===================================

# Modelo proposto: Modelo final

# Coleta dos metadados do dataset para o padrão do mlflow
dataset = mlflow.data.from_pandas(df_tempodist)
# Fórmula que serão utilizada no modelo
formula = "tempo ~ distancia"

# %%
# Criando o nosso primeiro experimento no mlflow
with mlflow.start_run(run_name="Modelo Final"):

    # =============================== INPUTS ==================================

    # Coleta dos metadados do dataset. O contexto pode ser utilizado para adi-
    # cionar mais uma informação de contexto do dataset, por exemplo: se ele é
    # um dataset de treino ou teste
    mlflow.log_input(dataset, context="training")

    # Parâmetros de entrada para serem coletados
    mlflow.log_param("Fórmula", formula)

    # ============================= MODELAGEM =================================

    # Treinamento do modelo
    modelo_final = sm.OLS.from_formula(
        formula=formula, data=df_tempodist
    ).fit()

    # ============================= OUTPUTS ===================================

    # Registrando as métricas do modelo
    mlflow.log_metric("Estatística F", modelo_final.fvalue)
    mlflow.log_metric("F p-value", modelo_final.f_pvalue)
    mlflow.log_metric("R2", modelo_final.rsquared)

    # Coletando artefatos do modelo

    # Gráfico do comparativo dos dados observados versus fittedvalues
    fig, ax = plt.subplots()
    ax.scatter(df_tempodist["tempo"], modelo_final.fittedvalues)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", color="gray")

    # Registro do gráfico como um artefato do experimento
    mlflow.log_figure(
        fig, "observado_vs_fitted.png"
    )  # importante definir a extensão

    # Obtendo o sumário do modelo em formato de texto
    modelo_final_summary_texto = modelo_final.summary().as_text()

    # Registro do sumário como um artefato do experimento
    mlflow.log_text(
        modelo_nulo_summary_texto, "summary.txt"
    )  # importante definir a extensão do arquivo

    # Carregamento do modelo no experimento
    mlflow.statsmodels.log_model(modelo_final, "modelo-final")

# %%

# Registrando uma nova versão do modelo tempo-dist
# Agora o modelo tempo-dist terá duas versões, o modelo nulo na versão 1 e o
# modelo final na versão 2.

# %%

# Após promover o modelo final para a fase de 'staging', o modelo nulo automa-
# ticamente o modelo anterior será e o novo modelo ficará registrado no estágio
# de 'staging' (adaptação)

# %%

# Carregando novamente o modelo tempo-dist na fase 'staging'
modelo_fase_staging = mlflow.statsmodels.load_model(
    "models:/tempo-distancia/staging"
)

# %%

# Verificando o R2
modelo_fase_staging.rsquared

# %%

# Verificando os fittedvalues
modelo_fase_staging.fittedvalues

# %%

# Realizando um predict
predict_novo_modelo_staging = modelo_fase_staging.predict(
    pd.DataFrame({"distancia": [20]})
)

# Verificando o predict
print(predict_novo_modelo_staging)

# %%

# ==================== GESTÃO DO CICLO DE VIDA DOS MODELOS ====================

# Ciclo esperado: Adatação -> Produção -> Arquivamento

# Promovendo o modelo final de fase de adaptação para a fase de produção
# Agora a versão 2 do modelo (modelo final) está em produção

# %%

# Carregando modelo tempo-dist em produção (production)
modelo_em_producao = mlflow.statsmodels.load_model(
    "models:/tempo-distancia/production"
)

# %%

# ===================== ENTENDENDO E CONSUMINDO APIS ==========================

# Alguns métodos HTTP bastante utilizados em requisições de APIs:

# GET: recupera dados do servidor
# POST: envia dados ao servidor para processamento.

fatos_cachorros = requests.get(
    "https://dogapi.dog/api/v2/facts",
    headers={"Content-Type": "application/json"},
)

# %%

# Coletando a resposta da requisição em formaton JSON
fatos_cachorros = fatos_cachorros.json()

# %%

# Acessando os campos e valores da resposta da API
print(fatos_cachorros["data"][0]["attributes"]["body"])

# %%

# =========== SERVINDO MODELOS EM PRODUÇÃO USANDO A API DO MLFLOW =============

# Acesse um novo terminal e ative o ambiente virtual que foi criado. Não feche
# o terminal que está rodando o Mlflow.
# Verifique o Passo 2 do "Instruções Aula - Big Data e Deployment de Modelos"
# para mais dúvidas.

# Para acessar um novo ambiente virtual
# Digite no terminal (no mesmo diretório onde a pasta venv se encontra):
# =============================================================================
# cd <DIRETORIO_ONDE_O_VENV_FOI_CRIADO>
# Windows: .\venv\Scripts\activate
# Linux/macOS: source /venv/bin/activate
# O nome (venv) deverá aparecer no início do campo do terminal

# Adicione a variável de ambiente para o rastreamento de onde está o Mlflow
# Digite no terminal:
# =============================================================================
# Linux/macOS: EXPORT MLFLOW_TRACKING_URI=http://localhost:5000
# Windows: SET MLFLOW_TRACKING_URI=http://localhost:5000
# =============================================================================

# Sirva o modelo tempo-distancia em fase de produção para ser acessado na
# porta 5200 via API do Mlflow
# Digite no terminal:
# =============================================================================
# mlflow models serve -m models:/tempo-distancia/production -p 5200 --no-conda
# =============================================================================

# %%

# Criando um dataframe para realizar um predict no nosso modelo em produção via
# API do Mlflow
df_novos_dados = pd.DataFrame({"distancia": [20]})

# %%

# Transformando os dados para o formato requisitação pela documentação da API
#  do mlflow
dados_transformados = json.dumps(
    {"dataframe_records": df_novos_dados.to_dict(orient="records")}
)

# %%

# Realizando a requisição ao modelo via API do Mlflow e coletando a resposta
response = requests.post(
    #url="http://localhost:5200/invocations",
    url="http://127.0.0.1:5200/invocations",
    data=dados_transformados,
    headers={"Content-Type": "application/json"},
)

# %%

# Verificando o retorno em formato de texto
response.text

# %%

# Verificando o retorno em formato JSON, que é compatível com o formato dicio-
# nário Python
predict = response.json()

# %%

# Acessando o valor do predict
print(predict["predictions"][0]["0"])

# %%

# =================== DEPLOYMENT DE UM MODELO HLM2 ============================

# Coletando o caminho do dataset
caminho = "./datasets/estudante_escola.csv"

# Carregando o dataset
df_estudante_escola = pd.read_csv(caminho)

# %%

# Relembrando o dataset df_estudante_escola
df_estudante_escola

# %%

# Verificando as informações do dataset e possíveis valores nulos
df_estudante_escola.info()

# %%

# Verificando as variáveis descritivas das colunas
df_estudante_escola[["escola", "desempenho"]].groupby(
    by=["escola"]
).describe()

# %%

# Criando o experimento para receber os modelos
mlflow.set_experiment(experiment_name="Desempenho de estudantes")

# %%

# ====================== PRIMEIRO MODELO DO EXPERIMENTO =======================

# Modelo proposto: Modelo nulo OLS

# Coleta dos metadados do dataset para o padrão do mlflow
dataset = mlflow.data.from_pandas(df_estudante_escola)
# Fórmula que serão utilizada no modelo
formula = "desempenho ~ 1"

# %%

# Permite que o Mlflow colete os logs (parámetros, métricas e artefatos) auto-
# maticamente
mlflow.statsmodels.autolog(
    log_models=True,  # registro do modelo
    log_datasets=True,  # registro dos metadados do dataset
    disable=False,  # habilitar o autolog
    registered_model_name=None,  # adicionar um nome default para o modelo
)

# Para desabilitar o autolog
# mlflow.statsmodels.autolog(disable=True)

# %%

# Criando o nosso primeiro experimento no mlflow
with mlflow.start_run(run_name="Modelo Nulo"):

    # ==================== INPUTS =============================================

    # Parâmetros de entrada e os metadados do dataset são coletadas de forma
    # automática por conta do comando mlflow.statsmodels.autolog()

    # ============================= MODELAGEM =================================

    # Treinamento do modelo
    modelo_estudante_escola_nulo = sm.OLS.from_formula(
        formula=formula, data=df_estudante_escola
    ).fit()

    # ==================== OUTPUTS ============================================

    # Métricas e artefatos do modelo são coletadas de forma automática por con-
    # ta do comando mlflow.statsmodels.autolog()

# %%

# ====================== SEGUNDO MODELO DO EXPERIMENTO ========================

# Modelo proposto: Modelo nulo HLM2

# Fórmula que serão utilizada no modelo
grupo = "escola"
componente_fixo = "desempenho ~ 1"
componente_aleatorio = "1"

# %%

# Criando o nosso primeiro experimento no mlflow
with mlflow.start_run(run_name="Modelo Nulo HLM2"):

    # ==================== INPUTS =============================================

    # Parâmetros de entrada e os metadados do dataset são coletadas de forma
    # automática por conta do comando mlflow.statsmodels.autolog()

    # ============================= MODELAGEM =================================

    # Treinamento do modelo
    modelo_estudante_escola_nulo_hlm2 = sm.MixedLM.from_formula(
        formula=componente_fixo,
        groups=grupo,
        re_formula=componente_aleatorio,
        data=df_estudante_escola,
    ).fit()

    # ==================== OUTPUTS ============================================

    # Métricas e artefatos do modelo são coletadas de forma automática por con-
    # ta do comando mlflow.statsmodels.autolog()

# %%

# ====================== TERCEIRO MODELO DO EXPERIMENTO =======================

# Modelo proposto: Modelo com interceptos aleatórios HLM2

# Fórmula que serão utilizada no modelo
grupo = "escola"
componente_fixo = "desempenho ~ horas"
componente_aleatorio = "1"

# %%

# Criando o nosso primeiro experimento no mlflow
with mlflow.start_run(run_name="Modelo com Interceptos Aleatórios HLM2"):

    # ==================== INPUTS =============================================

    # Parâmetros de entrada e os metadados do dataset são coletadas de forma
    # automática por conta do comando mlflow.statsmodels.autolog()

    # ============================= MODELAGEM =================================

    # Treinamento do modelo
    modelo_estudante_escola_intercept_hlm2 = sm.MixedLM.from_formula(
        formula=componente_fixo,
        groups=grupo,
        re_formula=componente_aleatorio,
        data=df_estudante_escola,
    ).fit()

    # ==================== OUTPUTS ============================================

    # Métricas e artefatos do modelo são coletadas de forma automática por con-
    # ta do comando mlflow.statsmodels.autolog()

# %%

# ====================== QUARTO MODELO DO EXPERIMENTO =======================

# Modelo proposto: Modelo com interceptos e inclinações aleatórios HLM2

# Fórmula que serão utilizada no modelo
grupo = "escola"
componente_fixo = "desempenho ~ horas"
componente_aleatorio = "horas"

# %%

# Criando o nosso primeiro experimento no mlflow
with mlflow.start_run(
    run_name="Modelo com Interceptos e Inclinações Aleatórios HLM2"
):

    # ==================== INPUTS =============================================

    # Parâmetros de entrada e os metadados do dataset são coletadas de forma
    # automática por conta do comando mlflow.statsmodels.autolog()

    # ============================= MODELAGEM =================================

    # Treinamento do modelo
    modelo_estudante_escola_intercept_inclin_hlm2 = sm.MixedLM.from_formula(
        formula=componente_fixo,
        groups=grupo,
        re_formula=componente_aleatorio,
        data=df_estudante_escola,
    ).fit()

    # ==================== OUTPUTS ============================================

    # Métricas e artefatos do modelo são coletadas de forma automática por con-
    # ta do comando mlflow.statsmodels.autolog()

# %%

# ====================== QUNTO MODELO DO EXPERIMENTO ==========================

# Modelo proposto: Modelo final com interceptos e inclinações aleatórios HLM2

# Fórmula que serão utilizada no modelo
grupo = "escola"
componente_fixo = "desempenho ~ horas + texp + horas:texp"
componente_aleatorio = "horas"

# %%

# Criando o nosso primeiro experimento no mlflow
with mlflow.start_run(
    run_name="Modelo Final com Interceptos e Inclinações Aleatórios HLM2"
):

    # ==================== INPUTS =============================================

    # Parâmetros de entrada e os metadados do dataset são coletadas de forma
    # automática por conta do comando mlflow.statsmodels.autolog()

    # ============================= MODELAGEM =================================

    # Treinamento do modelo
    modelo_estudante_escola_final_hlm2 = sm.MixedLM.from_formula(
        formula=componente_fixo,
        groups=grupo,
        re_formula=componente_aleatorio,
        data=df_estudante_escola,
    ).fit()

    # ==================== OUTPUTS ============================================

    # Métricas e artefatos do modelo são coletadas de forma automática por con-
    # ta do comando mlflow.statsmodels.autolog()

# %%

# Registrando o modelo para a fase de produção para ser consumido via API do
# Mlflow

# %%

# =================== SERVINDO O MODELO HLM2 EM PRODUÇÃO  =====================

# Acesse um novo terminal e ative o ambiente virtual que foi criado. Não feche
# o terminal que está rodando o Mlflow.
# Verifique o Passo 2 do "Instruções Aula - Big Data e Deployment de Modelos"
# para mais dúvidas.

# Para acessar um novo ambiente virtual
# Digite no terminal (no mesmo diretório onde a pasta venv se encontra):
# =============================================================================
# cd <DIRETORIO_ONDE_O_VENV_FOI_CRIADO>
# Windows: .\venv\Scripts\activate
# Linux/macOS: source /venv/bin/activate
# O nome (venv) deverá aparecer no início do campo do terminal

# Adicione a variável de ambiente para o rastreamento de onde está o Mlflow
# Digite no terminal:
# =============================================================================
# Linux/macOS: EXPORT MLFLOW_TRACKING_URI=http://localhost:5000
# Windows: SET MLFLOW_TRACKING_URI=http://localhost:5000
# =============================================================================

# Sirva o modelo estudante-escola em fase de produção para ser acessado na
# porta 5300 via API do Mlflow
# Digite no terminal:
# =============================================================================
# mlflow models serve -m models:/estudante-escola/production -p 5300 --no-conda
# =============================================================================

# %%

# Criando um dataframe para realizar um predict no nosso modelo em produção via
# API do Mlflow
df_novos_dados = pd.DataFrame({"escola": ["1"], "horas": [11], "texp": [3.6]})

# Transformando os dados para o formato requisitação pela documentação da API
#  do mlflow
dados_transformados = json.dumps(
    {"dataframe_records": df_novos_dados.to_dict(orient="records")}
)

# %%

# Realizando a requisição ao modelo via API do Mlflow e coletando a resposta
response = requests.post(
    #url="http://localhost:5200/invocations",
    url="http://127.0.0.1:5200/invocations",
    data=dados_transformados,
    headers={"Content-Type": "application/json"},
)

# %%

# Verificando o retorno da requisição
response.json()

# ERRO -> A Porta correta é a porta 5300, a porta 5200 está servindo o modelo


# %%

response = requests.post(
    #url="http://localhost:5300/invocations",
    url="http://127.0.0.1:5300/invocations",
    data=dados_transformados,
    headers={"Content-Type": "application/json"},
)

# %%

# Verificando o retorno da requisição
response.json()

# Lembrando que o predict da função MixedLM retorna o resultado do componente
# fixo

# %%

# Coletando o predict dentro do objeto json
response.json()['predictions'][0]["0"]

# %%

# =========================== FIM DO SCRIPT ===================================
