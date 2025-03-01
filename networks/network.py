import numpy as np
import networks.models as modeling
from networks import Mode
from src.tools import ValidationDatas, CreatePipeline
from src.normalize import NormalizeNum, NormalizeCat
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
from src import Aprendizado
from scikeras.wrappers import KerasClassifier, KerasRegressor
import polars as pl

CHUNK_SIZE = 10000

def train_model(reader, config_model):
    pipeline, y_test, X_test, y_true, X_original, dados_validation, nomes_colunas = run_training(reader, config_model)

    labels = None
    if config_model['mode']['learning_network'] == Aprendizado.UNSUPERVISED:
        # Atribui cada ponto de dados a um neurônio vencedor (cluster)
        winners = np.array([pipeline.winner(x) for x in X_test])
        # Converte as coordenadas do neurônio vencedor em um único rótulo de cluster
        labels = np.ravel_multi_index(winners.T, (config_model['mode']['params']['map_size_1'], config_model['mode']['params']['map_size_2']))

    calculate_statics(config_model, pipeline, y_test, X_test, y_true, X_original, labels, dados_validation, nomes_colunas)

    # Salvar o modelo
    joblib.dump(pipeline, f"model_{str(config_model['mode']['algorithm']).lower()}_{str(config_model['mode']['name_mode']).lower()}_{str(config_model['learning']).lower()}_{str(config_model['mode']['learning_network']).lower()}_ref{datetime.now().strftime("%Y%m%d")}.pkl")
    print('Modelo salvo com sucesso!')

def run_training(reader, config_model):
    # Variáveis para acompanhar o progresso do treinamento
    chunk_count = 0
    total_samples = 0

    # Listas para armazenar os dados de validação
    y_test = []
    X_test = []
    X_original = []
    y_true = []

    pipeline = None
    pip = None
    target = None

    print("Lendo os dados")
    # Lê o arquivo
    df_file_read = reader.read_data()
    nomes_colunas = None

    if config_model['mode']['learning_network'] == Aprendizado.SUPERVISED:

        if config_model['mode']['target'] is not None:
            target = config_model['mode']['target']
        else:
            raise ValueError(f"É necessário passar qual é o campo/coluna 'target'. Você enviou: {config_model['mode']['target']}")
            exit()

        num_epochs = None
        batch_size = None

        if config_model['mode']['params']['power']:
            num_epochs = config_model['mode']['params']['epoch']
            batch_size = config_model['mode']['params']['batch_size']

        X = df_file_read.drop(target) # Recursos (features)
        y = df_file_read[target]  # Rótulos (target)
        nomes_colunas = X.columns
        nome_target = y.to_frame().columns

        print("Configurando o Modelo")
        pipeline = CreatePipeline().choose_model_ml(config_model, modeling=modeling)
        pip = pipeline

        print("Separação dos dados em treino e validação")
        # Separação dos dados em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treina o modelo
        print("Treinando o modelo")
        if config_model['mode']['params']['power']:
            pipeline = pipeline.named_steps['model']
            pipeline.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)
        else:
            pipeline.fit(X_train, y_train)

        # Armazena os dados de validação
        X_test.append(X_val)  # Dados desconhecido pelo o treinamento
        y_test.append(y_val)  # Rotulos desconhecido

    elif config_model['mode']['learning_network'] == Aprendizado.UNSUPERVISED:
        X = None
        y = None
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

        print("Configurando o Modelo")
        config_model['mode']['params']['input_len'] = X.shape[1]
        pipeline = CreatePipeline().choose_model_ml(config_model, modeling=modeling)

        # Normalizar dados de forma diferente para a rede SOM
        if 'preprocessor' in dict(pipeline.steps):
            print("Normalizando os dados")
            preprocessor = pipeline.named_steps['preprocessor']

            X = pl.DataFrame(preprocessor.fit_transform(X), schema=nomes_colunas)
            if y is not None:
                colum_target = y.to_frame().columns

                num = None
                cat = None
                norma_num = None
                norma_cat = None

                if y.to_frame()[(colum_target[0])].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
                    num = colum_target
                    norma_num = config_model['pipeline']['normalize_method_num']

                if y.to_frame()[colum_target[0]].dtype in [pl.Utf8, pl.Categorical]:
                    cat = colum_target
                    norma_cat = config_model['pipeline']['normalize_method_cat']

                preprocessor = CreatePipeline().create_preprocessing_pipeline(numeric_features=num,
                                                                          categorical_features=cat, normalize_method_num= norma_num,
                                                                          normalize_method_cat=norma_cat)

                y = pl.DataFrame(preprocessor.fit_transform(y.to_frame()), schema=colum_target)

        pip = pipeline
        pipeline = pipeline.named_steps['model']

        print("Separação dos dados em treino e validação")
        # Separação dos dados em treino e validação
        X_train, X_val, y_train, y_val = None, None, None, None
        if target is not None:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

        print("Treinando o modelo")
        pipeline.random_weights_init(X_train.to_numpy())
        num_epochs = config_model['mode']['params']['epoch']  # Número de épocas de treinamento

        # Treina o modelo
        pipeline.train_random(X_train.to_numpy(), num_iteration=num_epochs * len(X_train.to_numpy()), verbose=False)  # Número total de iterações

        # Armazena os dados de validação
        X_test.append(X_val)
        X_original.append(X)  # Acumula os dados originais
        if target is not None:
            y_true.append(y)  # Acumula os rótulos verdadeiros
        else:
            y_true = None

    else:
        raise ValueError(f"Não foi encontrando nenhum tipo de aprendizado para: {config_model['mode']['learning_network']}")
        exit()

    if config_model['mode']['learning_network'] == Aprendizado.SUPERVISED:
        # Concatena todos os dados de validação
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

    if config_model['mode']['learning_network'] == Aprendizado.UNSUPERVISED:
        # Concatena todos os dados de validação
        X_test = np.concatenate(X_test)
        X_original = np.concatenate(X_original)
        if target is not None:
            y_true = np.concatenate(y_true)
        else:
            y_true = None

    dados_validation = None
    if config_model['validations'] is not None:
        if config_model['mode']['learning_network'] == Aprendizado.SUPERVISED:
            pipeline, dados_validation = ValidationDatas().get_validations(config_model, pip, X_test, y_test)
        elif config_model['mode']['learning_network'] == Aprendizado.UNSUPERVISED:
            pipeline, dados_validation = ValidationDatas().get_validations(config_model, pip, X_original, y_true)
            pipeline = pipeline.named_steps['model']

    return pipeline, y_test, X_test, y_true, X_original, dados_validation, nomes_colunas

def calculate_statics(config_model, pipeline, y_test, X_test, y_true, X_original, labels, dados_validation, nomes_colunas):
    modeling.statics(config_model, pipeline, y_test, X_test, y_true, X_original, labels, dados_validation, nomes_colunas)
