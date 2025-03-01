import numpy as np
import semisupervised.models as modeling
from src.tools import ValidationDatas, CreatePipeline
from src.normalize import NormalizeNum, NormalizeCat
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
import joblib
from datetime import datetime

CHUNK_SIZE = 10000

def train_model(reader, config_model):
    pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas = run_training(reader, config_model)
    calculate_statics(config_model, pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas)

    # Salvar o modelo
    joblib.dump(pipeline, f"model_{str(config_model['mode']['algorithm']).lower()}_{str(config_model['mode']['name_mode']).lower()}_{str(config_model['learning']).lower()}_ref{datetime.now().strftime("%Y%m%d")}.pkl")
    print('Modelo salvo com sucesso!')

def run_training(reader, config_model):
    # Variáveis para acompanhar o progresso do treinamento
    chunk_count = 0
    total_samples = 0

    # Listas para armazenar os dados de validação
    y_test = []
    X_test = []

    print("Lendo os dados")
    # Lê o arquivo
    df_file_read = reader.read_data()
    nomes_colunas = None

    target = None
    if config_model['mode']['target'] is not None:
        target = config_model['mode']['target']
    else:
        raise ValueError(f"É necessário passar qual é o campo/coluna 'target'. Você enviou: {config_model['mode']['target']}")
        exit()

    X = df_file_read.drop(target)  # Recursos (features)
    y = df_file_read[target]  # Rótulos (target)
    nomes_colunas = X.columns

    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y.to_numpy())) < 0.8  # 80% dos dados não rotulados
    y = np.copy(y)
    y[random_unlabeled_points] = -1  # Rótulos desconhecidos são marcados como -1

    print("Configurando o Modelo")
    pipeline = CreatePipeline().choose_model_ml(config_model, modeling=modeling)

    print("Separação dos dados em treino e validação")
    # Separação dos dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Treinando o modelo")
    # Treina o modelo (apenas com os dados de treino)
    pipeline.fit(X_train, y_train)

    # Armazena os dados de validação
    X_test.append(X_val) #Dados do treinamento
    y_test.append(y_val) # Rótulos verdadeiros para os dados não rotulados

    # Concatena todos os dados de validação
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    dados_validation = None
    if config_model['validations'] is not None:
        pipeline, dados_validation = ValidationDatas().get_validations(config_model, pipeline, X_test, y_test)

    return pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas

def calculate_statics(config_model, pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas):
    modeling.statics(config_model, pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas)
