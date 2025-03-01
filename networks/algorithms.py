from sklearn.linear_model import Perceptron #Perceptron
from sklearn.neural_network import MLPClassifier #Multi-layer Perceptron (MLP)

from sklearn.neural_network import MLPRegressor #Multi-layer Perceptron (MLP) para Regressão

from minisom import MiniSom #Não supervisionado

#Redes mais poderosas
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness
from scipy.stats import f_oneway

from networks import Redes
import matplotlib.pyplot as plt
from datetime import datetime
from src.report import FileWrite
import numpy as np
import polars as pl

def get_algorithm_clas(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Redes.PERCEPTRON:
        #penalty - Regularização
        #alpha - Constante de regularização
        #max_iter - Número máximo de iterações
        #tol - Tolerância para parada
        #eta0 - Taxa de aprendizado inicial
        #shuffle - Embaralha os dados após cada época
        #random_state - Semente para reprodutibilidade
        model = Perceptron(penalty=config_model['penalty'], alpha=config_model['alpha'], max_iter=config_model['max_iter'], tol=config_model['tol'], eta0=config_model['eta0'], shuffle=config_model['shuffle'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Redes.MLP:
        # hidden_layer_sizes - Uma camada oculta com neurônios
        # activation - Função de ativação
        # solver - Algoritmo de otimização
        # alpha - Constante de regularização
        # learning_rate - Taxa de aprendizado
        # max_iter - Número máximo de iterações
        # tol - Tolerância para parada
        # random_state - Semente para reprodutibilidade
        model = MLPClassifier(hidden_layer_sizes=config_model['hidden_layer_sizes'], activation=config_model['activation'], solver=config_model['solver'], alpha=config_model['alpha'], learning_rate=config_model['learning_rate'], max_iter=config_model['max_iter'], tol=config_model['tol'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Redes.DNN:
        # n_camadas - numero de camadas
        # activation_enter - Função de ativação da camada de entrada
        # n_neuronio_enter - Numero de neuronios de entrada, geralmente é o numero de dados que são passados
        # activation_intermediate - Função de ativação da camada intermediária
        # n_neuronio_intermediate -Numero de neuronio da camada intermediária
        # activation_end - Função de ativação da camada de saida
        # n_neuronio_end - Numero de neuronios de saida, geralmente é o numero de classes
        #input_shape - Forma dos dados de entrada (tupla de inteiros)
        #optimizer - Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        #loss - Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        #metrics - Métricas para avaliação (lista de strings - exemplo: accuracy)
        model = keras.Sequential()
        model.add(keras.layers.Dense(config_model['n_neuronio_enter'], activation=config_model['activation_enter'], input_shape=config_model['input_shape']))  # Camada de entrada
        for _ in range(config_model['n_camadas']):
            model.add(keras.layers.Dense(config_model['n_neuronio_intermediate'], activation=config_model['activation_intermediate'])) # Camada intermediária
        model.add(keras.layers.Dense(config_model['n_neuronio_end'], activation=config_model['activation_end']))  # Camada de saída

        model.compile(optimizer=config_model['optimizer'], loss=config_model['loss'], metrics=config_model['metrics'])
    elif str(config_model_job['algorithm']).lower() == Redes.CNN:
        # n_camadas - numero de camadas
        # activation_enter - Função de ativação da camada de entrada
        # n_neuronio_enter - Numero de neuronios de entrada, geralmente é o numero de dados que são passados
        # activation_intermediate - Função de ativação da camada intermediária
        # n_neuronio_intermediate -Numero de neuronio da camada intermediária
        # activation_end - Função de ativação da camada de saida
        # n_neuronio_end - Numero de neuronios de saida, geralmente é o numero de classes
        # input_shape - Forma dos dados de entrada (tupla de inteiros: altura, largura, canais)
        # kernel_enter - Tamanho do kernel na camada de entrada (tupla de inteiros)
        # kernel_intermediate - Tamanho do kernel nas camadas intermediárias (tupla de inteiros)
        # pooling_enter - Tamanho do pooling na camada de entrada (tupla de inteiros)
        # pooling_intermediate - Tamanho do pooling nas camadas intermediárias (tupla de inteiros)
        # optimizer - Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        # loss - Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        # metrics - Métricas para avaliação (lista de strings - exemplo: accuracy)
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(config_model['n_neuronio_enter'], config_model['kernel_enter'], activation=config_model['activation_enter'], input_shape=config_model['input_shape']))
        model.add(keras.layers.MaxPooling2D(config_model['pooling_enter']))

        for _ in range(config_model['n_camadas'] - 1):  # A primeira camada já foi adicionada
            model.add(keras.layers.Conv2D(config_model['n_neuronio_intermediate'], config_model['kernel_intermediate'], activation=config_model['activation_intermediate']))
            model.add(keras.layers.MaxPooling2D(config_model['pooling_intermediate']))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(config_model['n_neuronio_intermediate']*2, activation=config_model['activation_intermediate']))
        model.add(keras.layers.Dense(config_model['n_neuronio_end'], activation=config_model['activation_end']))

        model.compile(optimizer=config_model['optimizer'], loss=config_model['loss'], metrics=config_model['metrics'])
    elif str(config_model_job['algorithm']).lower() == Redes.RNN:
        # n_camadas - numero de camadas
        # return_sequences_enter - Retornar sequências na camada de entrada (booleano: True ou False)
        # return_sequences_intermediate - Retornar sequências nas camadas intermediárias (booleano: True ou False)
        # activation_enter - Função de ativação da camada de entrada (string: 'tanh', 'relu', 'sigmoid', etc.)
        # n_neuronio_enter - Numero de neuronios de entrada, geralmente é o numero de dados que são passados
        # activation_intermediate - Função de ativação da camada intermediária
        # n_neuronio_intermediate -Numero de neuronio da camada intermediária
        # activation_end - Função de ativação da camada de saida
        # n_neuronio_end - Numero de neuronios de saida, geralmente é o numero de classes
        # input_shape - Forma dos dados de entrada (tupla de inteiros: passos de tempo, características)
        # optimizer - Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        # loss - Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        # metrics - Métricas para avaliação (lista de strings - exemplo: accuracy)
        model = keras.Sequential()
        if config_model['n_camadas'] > 0:
            model.add(keras.layers.LSTM(config_model['n_neuronio_enter'], return_sequences=config_model['return_sequences_enter'], input_shape=config_model['input_shape']))

            for _ in range(config_model['n_camadas'] - 2):  # -2 pois a primeira e a ultima camada são tratadas separadamente.
                model.add(keras.layers.LSTM(config_model['n_neuronio_intermediate'], return_sequences=config_model['return_sequences_intermediate']))

            if config_model['n_camadas'] > 1:
                model.add(keras.layers.LSTM(config_model['n_neuronio_intermediate']))
            else:
                model.add(keras.layers.LSTM(config_model['n_neuronio_intermediate'], return_sequences=(not config_model['return_sequences_intermediate'])))

        model.add(keras.layers.Dense(config_model['n_neuronio_end'], activation=config_model['activation_end']))

        model.compile(optimizer=config_model['optimizer'], loss=config_model['loss'], metrics=config_model['metrics'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def get_algorithm_reg(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Redes.MLP:
        # hidden_layer_sizes - Uma camada oculta com neurônios
        # activation - Função de ativação
        # solver - Algoritmo de otimização
        # alpha - Constante de regularização
        # learning_rate - Taxa de aprendizado
        # max_iter - Número máximo de iterações
        # tol - Tolerância para parada
        # random_state - Semente para reprodutibilidade
        model = MLPRegressor(hidden_layer_sizes=config_model['hidden_layer_sizes'], activation=config_model['activation'], solver=config_model['solver'], alpha=config_model['alpha'], learning_rate=config_model['learning_rate'], max_iter=config_model['max_iter'], tol=config_model['tol'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Redes.DNN:
        #n_camadas - numero de camadas
        # activation_enter - Função de ativação da camada de entrada
        # n_neuronio_enter - Numero de neuronios de entrada, geralmente é o numero de dados que são passados
        # activation_intermediate - Função de ativação da camada intermediária
        # n_neuronio_intermediate -Numero de neuronio da camada intermediária
        # activation_end - Função de ativação da camada de saida
        # n_neuronio_end - Numero de neuronios de saida, geralmente é o numero de classes
        # input_shape - Forma dos dados de entrada (tupla de inteiros)
        # optimizer - Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        # loss - Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        # metrics - Métricas para avaliação (lista de strings - exemplo: accuracy)
        model = keras.Sequential()
        model.add(keras.layers.Dense(config_model['n_neuronio_enter'], activation=config_model['activation_enter'], input_shape=config_model['input_shape']))  # Camada de entrada
        for _ in range(config_model['n_camadas']):
            model.add(keras.layers.Dense(config_model['n_neuronio_intermediate'], activation=config_model['activation_intermediate'])) # Camada intermediária
        model.add(keras.layers.Dense(config_model['n_neuronio_end'], activation=config_model['activation_end']))  # Camada de saída

        model.compile(optimizer=config_model['optimizer'], loss=config_model['loss'], metrics=config_model['metrics'])
    elif str(config_model_job['algorithm']).lower() == Redes.CNN:
        # n_camadas - numero de camadas
        # activation_enter - Função de ativação da camada de entrada
        # n_neuronio_enter - Numero de neuronios de entrada, geralmente é o numero de dados que são passados
        # activation_intermediate - Função de ativação da camada intermediária
        # n_neuronio_intermediate -Numero de neuronio da camada intermediária
        # activation_end - Função de ativação da camada de saida
        # n_neuronio_end - Numero de neuronios de saida, geralmente é o numero de classes
        # input_shape - Forma dos dados de entrada (tupla de inteiros: altura, largura, canais)
        # kernel_enter - Tamanho do kernel na camada de entrada (tupla de inteiros)
        # kernel_intermediate - Tamanho do kernel nas camadas intermediárias (tupla de inteiros)
        # pooling_enter - Tamanho do pooling na camada de entrada (tupla de inteiros)
        # pooling_intermediate - Tamanho do pooling nas camadas intermediárias (tupla de inteiros)
        # optimizer - Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        # loss - Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        # metrics - Métricas para avaliação (lista de strings - exemplo: accuracy)
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(config_model['n_neuronio_enter'], config_model['kernel_enter'], activation=config_model['activation_enter'], input_shape=config_model['input_shape']))
        model.add(keras.layers.MaxPooling2D(config_model['pooling_enter']))

        for _ in range(config_model['n_camadas'] - 1):  # A primeira camada já foi adicionada
            model.add(keras.layers.Conv2D(config_model['n_neuronio_intermediate'], config_model['kernel_intermediate'], activation=config_model['activation_intermediate']))
            model.add(keras.layers.MaxPooling2D(config_model['pooling_intermediate']))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(config_model['n_neuronio_intermediate']*2, activation=config_model['activation_intermediate']))
        model.add(keras.layers.Dense(config_model['n_neuronio_end'], activation=config_model['activation_end']))

        model.compile(optimizer=config_model['optimizer'], loss=config_model['loss'], metrics=config_model['metrics'])
    elif str(config_model_job['algorithm']).lower() == Redes.RNN:
        # n_camadas - numero de camadas
        # return_sequences_enter - Retornar sequências na camada de entrada (booleano: True ou False)
        # return_sequences_intermediate - Retornar sequências nas camadas intermediárias (booleano: True ou False)
        # activation_enter - Função de ativação da camada de entrada (string: 'tanh', 'relu', 'sigmoid', etc.)
        # n_neuronio_enter - Numero de neuronios de entrada, geralmente é o numero de dados que são passados
        # activation_intermediate - Função de ativação da camada intermediária
        # n_neuronio_intermediate -Numero de neuronio da camada intermediária
        # activation_end - Função de ativação da camada de saida
        # n_neuronio_end - Numero de neuronios de saida, geralmente é o numero de classes
        # input_shape - Forma dos dados de entrada (tupla de inteiros: passos de tempo, características)
        # optimizer - Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
        # loss - Função de perda (string: 'categorical_crossentropy', 'mse', 'binary_crossentropy', etc.)
        # metrics - Métricas para avaliação (lista de strings - exemplo: accuracy)
        model = keras.Sequential()
        if config_model['n_camadas'] > 0:
            model.add(keras.layers.LSTM(config_model['n_neuronio_enter'], return_sequences=config_model['return_sequences_enter'], input_shape=config_model['input_shape']))

            for _ in range(config_model['n_camadas'] - 2):  # -2 pois a primeira e a ultima camada são tratadas separadamente.
                model.add(keras.layers.LSTM(config_model['n_neuronio_intermediate'], return_sequences=config_model['return_sequences_intermediate']))

            if config_model['n_camadas'] > 1:
                model.add(keras.layers.LSTM(config_model['n_neuronio_intermediate']))
            else:
                model.add(keras.layers.LSTM(config_model['n_neuronio_intermediate'], return_sequences=(not config_model['return_sequences_intermediate'])))

        model.add(keras.layers.Dense(config_model['n_neuronio_end'], activation=config_model['activation_end']))

        model.compile(optimizer=config_model['optimizer'], loss=config_model['loss'], metrics=config_model['metrics'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def get_algorithm_clus(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Redes.SOM:
        #map_size - Tamanho do mapa (10x10 neurônios) (ex: (10, 10))
        #input_len - Número de features (dimensões dos dados) (ex: X.shape[1])
        #sigma - Raio inicial do vizinho
        #learning_rate - Taxa de aprendizado inicial
        #random_seed = Semente para reprodutibilidade
        model = MiniSom(x=config_model['map_size_1'], y=config_model['map_size_2'], input_len=config_model['input_len'], sigma=float(config_model['sigma']), learning_rate=float(config_model['learning_rate']), random_seed=config_model['random_seed'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def calculate_clas(pipeline, y_test, X_test, dados_validation, config_model, nomes_colunas):
    # Previsões
    X_test = pl.DataFrame(X_test, schema=nomes_colunas)
    pipeline = pipeline.named_steps['model']
    y_pred = pipeline.predict(X_test)
    if config_model['mode']['params']['power'] is True:
        y_pred = np.argmax(y_pred, axis=1)

    # Cálculo das métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracia: {accuracy}")
    print(f"Precisão: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Matriz de Confusão:\n{conf_matrix}")

    if accuracy >= 1.0:
        print("Possivel Overfitting")

    filename = f"NETWORK_SUPERVISED_CLASSFICATION_REF{datetime.now().strftime("%Y%m%d")}"
    FileWrite().save_file(filename=filename,
                        dados={"acuracia": accuracy, "precisao": precision, "recall": recall, "f1": f1,
                               "matriz_conf": conf_matrix},
                        type={"aprendizado": "supervised", "mode_aprendizado": "classificacao"},
                        validations=dados_validation)

def calculate_reg(pipeline, y_test, X_test, dados_validation, config_model):
    # Previsões
    pipeline = pipeline.named_steps['model']
    y_pred = pipeline.predict(X_test)
    if config_model['mode']['params']['power'] is True:
        y_pred = np.argmax(y_pred, axis=1)

    mse = None
    mae = None
    r2 = None
    # Cálculo das métricas
    if len(y_test) == len(y_pred):
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    else:
        mse = f"Não foi possivel calcular, pois seu modelo resultou em tamanhos diferentes: Dados de validação --> {len(y_test)}. Dados de previsão --> {len(y_pred)}."
        mae = f"Não foi possivel calcular, pois seu modelo resultou em tamanhos diferentes: Dados de validação --> {len(y_test)}. Dados de previsão --> {len(y_pred)}."
        r2 = f"Não foi possivel calcular, pois seu modelo resultou em tamanhos diferentes: Dados de validação --> {len(y_test)}. Dados de previsão --> {len(y_pred)}."

    print(f"Erro Quadrático Médio (MSE): {mse}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    print(f"Coeficiente de Determinação (R²): {r2}")

    filename = f"NETWORK_SUPERVISED_REGRESSION_REF{datetime.now().strftime("%Y%m%d")}"
    FileWrite().save_file(filename=filename,
                        dados={"mse": mse, "mae": mae, "r2": r2},
                        type={"aprendizado": "supervised", "mode_aprendizado": "regressao"},
                        validations=dados_validation)

def calculate_clus(pipeline, X_test, y_true=None, labels=None, X=None, dados_validation=None):
    silhouette = None
    calinski_harabasz = None
    davies_bouldin = None
    adjusted_rand = None
    normalized_mutual_info = None
    anova_resuts = None

    if (labels is not None and np.unique(labels).size > 1) and len(X_test) == len(labels):
        # Métricas intrínsecas
        silhouette = silhouette_score(X_test, labels)
        calinski_harabasz = calinski_harabasz_score(X_test, labels)
        davies_bouldin = davies_bouldin_score(X_test, labels)
        print("Silhouette Score:", silhouette)
        print("Calinski-Harabasz Index:", calinski_harabasz)
        print("Davies-Bouldin Index:", davies_bouldin)
    else:
        silhouette = "Não foi possivel calcular o Score de Silhouette"
        calinski_harabasz = "Não foi possivel calcular o Indice de Calinski-Harabasz"
        davies_bouldin = "Não foi possivel calcular o Indice de Davies-Bouldin"
        print(silhouette)
        print(calinski_harabasz)
        print(davies_bouldin)

    if ((labels is not None and y_true is not None) and np.unique(labels).size > 1) and len(y_true) == len(labels):
        # Métricas extrínsecas (se y_true estiver disponível)
        adjusted_rand = adjusted_rand_score(y_true, labels)
        normalized_mutual_info = normalized_mutual_info_score(y_true, labels)
        print("Adjusted Rand Index:", adjusted_rand)
        print("Normalized Mutual Information:", normalized_mutual_info)
    else:
        adjusted_rand = "Não foi possivel calcular o Indice de Adjusted Rand"
        normalized_mutual_info = "Não foi possivel calcular o Normalized Mutual Information (NMI)"
        print(adjusted_rand)
        print(normalized_mutual_info)

    #Calculo de ANOVA
    if (X_test is not None and labels is not None) and np.unique(labels).size > 1:
        anova_results = calculate_anova(X_test, labels)
        print(f"Resultados da ANOVA: {anova_results}")
    else:
        anova_results = "Não foi possivel calcular o ANOVA, pois faltam passar os rotulos obtidos e/ou dados originais"
        print(anova_results)

    # Visualiza os resultados
    plt.figure(figsize=(10, 10))
    plt.pcolor(pipeline.distance_map().T, cmap='bone_r')  # Distância entre neurônios
    plt.colorbar()

    filename = f"NETWORK_UNSUPERVISED_CLUSTERING_REF{datetime.now().strftime("%Y%m%d")}"
    # Plota os dados no mapa
    for i, x in enumerate(X):
        w = pipeline.winner(x)  # Encontra o neurônio vencedor para o dado x
        plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='red', markersize=10,
                 markeredgewidth=2)

    plt.title("Mapa de Kohonen (SOM)")
    name_fig = f'KOHONEM_{filename}.png'
    plt.savefig(name_fig)

    FileWrite().save_file(filename=filename,
                        dados={"silhouette_score": silhouette, "calinski_harabasz_score": calinski_harabasz, "davies_bouldin_score": davies_bouldin, "adjusted_rand_score":adjusted_rand, "normalized_mutual_info": normalized_mutual_info, "anova":anova_results},
                        type={"aprendizado": "unsupervised", "mode_aprendizado": "cluster", "mode_aprendizado_network":True, "mapa":name_fig},
                        validations=dados_validation)

def calculate_anova(X, labels):
    """
    Calcula a ANOVA para cada feature em relação aos clusters.

    :param X: Dados de entrada (numpy array ou polars DataFrame).
    :param labels: Rótulos dos clusters (numpy array).
    :return: DataFrame do polars com os resultados da ANOVA (F-statistic e p-value) para cada feature.
    """
    # Converte X para polars DataFrame (se ainda não for)
    if not isinstance(X, pl.DataFrame):
        X = pl.DataFrame(X)

    # Listas para armazenar os resultados
    features = X.columns
    f_stats = []
    p_values = []

    # Calcula a ANOVA para cada feature
    for feature in features:
        # Extrai os valores da feature e agrupa por cluster
        feature_values = X[feature].to_numpy()
        groups = [feature_values[labels == cluster] for cluster in np.unique(labels)]

        # Calcula a ANOVA
        f_stat, p_value = f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_value)

    # Cria um DataFrame do polars com os resultados
    anova_results = pl.DataFrame({
        'Feature': features,
        'F-statistic': f_stats,
        'p-value': p_values
    })

    return anova_results