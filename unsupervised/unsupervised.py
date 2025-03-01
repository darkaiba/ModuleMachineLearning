import numpy as np
import unsupervised.models as modeling
from src.tools import ValidationDatas, CreatePipeline
from src.normalize import NormalizeNum, NormalizeCat
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
from unsupervised import Mode
import polars as pl

CHUNK_SIZE = 10000

def train_model(reader, config_model):
    pipeline, X_test, y_true, X_original, dados_validation, nomes_colunas = run_training(reader, config_model)

    labels = None
    if str(config_model['mode']['name_mode']).lower() == Mode.CLUSTER:
        labels = []
        if len(X_test) > 1:  # Requer pelo menos 2 amostras
            labels = pipeline.predict(X_test)
    elif str(config_model['mode']['name_mode']).lower() == Mode.REDUCE:
        labels = []
        if len(X_original) > 1:  # Requer pelo menos 2 amostras
            X_original = pl.DataFrame(X_original, schema=nomes_colunas)
            labels = pipeline.fit_transform(X_original)

    calculate_statics(config_model, y_true, X_test, X_original, labels, pipeline, dados_validation, nomes_colunas)

    # Salvar o modelo
    joblib.dump(pipeline, f"model_{str(config_model['mode']['algorithm']).lower()}_{str(config_model['mode']['name_mode']).lower()}_{str(config_model['learning']).lower()}_ref{datetime.now().strftime("%Y%m%d")}.pkl")
    print('Modelo salvo com sucesso!')


def run_training(reader, config_model):
    # Variáveis para acompanhar o progresso do treinamento
    chunk_count = 0
    total_samples = 0

    # Listas para armazenar os dados de validação
    y_true = []
    X_test = []
    X_original = []

    print("Lendo os dados")
    # Lê o arquivo
    target = None
    df_file_read = reader.read_data()

    X = None
    y = None
    nomes_colunas = None
    if config_model['mode']['target'] is not None:
        target = config_model['mode']['target']

    if target is not None:
        print("Target foi passado para o Aprendizado Não Supervsionado!")
        X = df_file_read.drop('target')  # Recursos (features)
        y = df_file_read['target']  # Rótulos (target)
    else:
        print("Target não foi passado para o Aprendizado Não Supervsionado!")
        X = df_file_read
        y = None

    nomes_colunas = X.columns

    print("Configurando o Modelo")
    pipeline = CreatePipeline().choose_model_ml(config_model, modeling=modeling)

    print("Separação dos dados em treino e validação")
    # Separação dos dados em treino e validação
    X_train, X_val, y_train, y_val = None, None, None, None
    if target is not None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    print("Treinando o modelo")
    # Treina o modelo
    if target is not None:
        pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train)

    # Armazena os dados de validação
    X_test.append(X_val)# Acumula os dados de validação
    X_original.append(X_train)
    if target is not None:
        y_true.append(y_train)  # Acumula os rótulos verdadeiros se tiver
    else:
        y_true = None

    # Concatena todos os dados de validação
    X_test = np.concatenate(X_test)
    X_original = np.concatenate(X_original)
    if target is not None:
        y_true = np.concatenate(y_true)
    else:
        y_true = None

    dados_validation = None
    if config_model['validations'] is not None:
        pipeline, dados_validation = ValidationDatas().get_validations(config_model, pipeline, X_original, y_true)

    return pipeline, X_test, y_true, X_original, dados_validation, nomes_colunas

def calculate_statics(config_model, y_true, X_test, X_original, labels, pipeline, dados_validation, nomes_colunas):
    modeling.statics(config_model, y_true, X_test, X_original, labels, pipeline, dados_validation, nomes_colunas)