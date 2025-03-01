from sklearn.linear_model import LogisticRegression #Regressão Logistica
from sklearn.svm import SVC #Máquinas de Vetores de Suporte (SVM)
from sklearn.neighbors import KNeighborsClassifier #k-Nearest Neighbors (k-NN)
from sklearn.tree import DecisionTreeClassifier #Árvores de Decisão
from sklearn.ensemble import RandomForestClassifier #Florestas Aleatórias
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosting
from sklearn.ensemble import AdaBoostClassifier #AdaBoost
from sklearn.naive_bayes import GaussianNB #Naive Bayes

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

from supervised import Classificador
from src.report import FileWrite
from datetime import datetime

from sklearn.linear_model import LinearRegression #Regressão Linear
from sklearn.linear_model import Ridge #Regressão Ridge
from sklearn.linear_model import Lasso #Regressão Lasso
from sklearn.linear_model import ElasticNet #Regressão ElasticNet
from sklearn.svm import SVR #SVM para Regressão
from sklearn.tree import DecisionTreeRegressor #Árvores de Decisão para Regressão
from sklearn.ensemble import RandomForestRegressor #Florestas Aleatórias para Regressão
from sklearn.ensemble import GradientBoostingRegressor #Gradient Boosting para Regressão
from sklearn.ensemble import AdaBoostRegressor #AdaBoost para Regressão

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from supervised import Regressor
from src.report import FileWrite
from datetime import datetime

import polars as pl

def get_algorithm_clas(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Classificador.REGRESSAO_LOGISTICA:
        #penalty - Tipo de regularização: 'l1', 'l2', 'elasticnet', 'none'
        #C - Inverso da força de regularização
        #solver - Algoritmo para otimização: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
        #max_iter - Número máximo de iterações
        #random_state - Semente para reproducibilidade
        model = LogisticRegression(penalty=config_model['penalty'], C=config_model['C'], solver=config_model['solver'], max_iter=config_model['max_iter'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Classificador.SVM:
        #C - Parâmetro de regularização
        #kernel - Tipo de kernel: 'linear', 'poly', 'rbf', 'sigmoid'
        #degree - Grau do kernel polinomial (apenas para kernel 'poly')
        #gamma - Coeficiente do kernel (apenas para kernels 'rbf', 'poly', 'sigmoid')
        #probability - Habilita o cálculo de probabilidades
        #random_state - Semente para reproducibilidade
        model = SVC(C=config_model['C'], kernel=config_model['kernel'], degree=config_model['degree'], gamma=config_model['gamma'], probability=config_model['probability'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Classificador.KNN:
        #n_neighbors - Número de vizinhos
        #weights - Peso dos vizinhos: 'uniform', 'distance'
        #algorithm - Algoritmo para calcular os vizinhos: 'auto', 'ball_tree', 'kd_tree', 'brute'
        #p - Parâmetro da distância de Minkowski (1 para Manhattan, 2 para Euclidiana)
        model = KNeighborsClassifier(n_neighbors=config_model['n_neighbors'], weights=config_model['weights'], algorithm=config_model['algorithm'], p=config_model['p'])
    elif str(config_model_job['algorithm']).lower() == Classificador.ARVORE_DECISAO:
        #criterion - Função para medir a qualidade de uma divisão: 'gini', 'entropy'
        #max_depth - Profundidade máxima da árvore
        #min_samples_split - Número mínimo de amostras necessárias para dividir um nó
        #min_samples_leaf - Número mínimo de amostras necessárias em uma folha
        #random_state - Semente para reproducibilidade
        model = DecisionTreeClassifier(criterion=config_model['criterion'], max_depth=config_model['max_depth'], min_samples_split=config_model['min_samples_split'], min_samples_leaf=config_model['min_samples_leaf'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Classificador.FLORESTA_ALEATORIA:
        #n_estimators - Número de árvores na floresta
        #criterion - Função para medir a qualidade de uma divisão: 'gini', 'entropy'
        #max_depth - Profundidade máxima de cada árvore
        #min_samples_split - Número mínimo de amostras necessárias para dividir um nó
        #min_samples_leaf - Número mínimo de amostras necessárias em uma folha
        #random_state - Semente para reproducibilidade
        model = RandomForestClassifier(n_estimators=config_model['n_estimators'], criterion=config_model['criterion'], max_depth=config_model['max_depth'], min_samples_split=config_model['min_samples_split'], min_samples_leaf=config_model['min_samples_leaf'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Classificador.GRADIENT_BOOST:
        #n_estimators - Número de estágios de boosting
        #learning_rate - Taxa de aprendizado
        #max_depth - Profundidade máxima de cada árvore
        #min_samples_split - Número mínimo de amostras necessárias para dividir um nó
        #min_samples_leaf - Número mínimo de amostras necessárias em uma folha
        #random_state - Semente para reproducibilidade
        model = GradientBoostingClassifier(n_estimators=config_model['n_estimators'], learning_rate=config_model['learning_rate'], max_depth=config_model['max_depth'], min_samples_split=config_model['min_samples_split'], min_samples_leaf=config_model['min_samples_leaf'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Classificador.ADA_BOOST:
        #n_estimators - Número de estimadores
        #learning_rate - Taxa de aprendizado
        #algorithm - Algoritmo: 'SAMME', 'SAMME.R'
        #random_state - Semente para reproducibilidade
        model = AdaBoostClassifier(n_estimators=config_model['n_estimators'], learning_rate=config_model['learning_rate'], algorithm=config_model['algorithm'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Classificador.NAIVE_BAYES:
        #priors - Probabilidades a priori das classes (se None, são calculadas a partir dos dados)
        #var_smoothing - Parâmetro de suavização para evitar variância zero
        model = GaussianNB(priors=config_model['priors'], var_smoothing=config_model['var_smoothing'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def get_algorithm_reg(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Regressor.REGRESSAO_LINEAR:
        #fit_intercept - Se True, calcula o intercepto (bias)
        #copy_X - Se True, copia os dados de entrada (evita modificação dos dados originais)
        #n_jobs - Número de jobs para paralelização (None = 1, -1 = todos os cores)
        model = LinearRegression(fit_intercept=config_model['fit_intercept'], copy_X=config_model['copy_X'], n_jobs=config_model['n_jobs'])
    elif str(config_model_job['algorithm']).lower() == Regressor.REGRESSAO_RIDGE:
        #alpha - Parâmetro de regularização (quanto maior, mais forte a regularização)
        #fit_intercept - Se True, calcula o intercepto (bias)
        #copy_X - Se True, copia os dados de entrada
        #max_iter - Número máximo de iterações para o solver
        #tol - Tolerância para critério de parada
        #solver - Algoritmo de otimização: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
        #random_state - Semente para reproducibilidade (usado em 'sag' e 'saga')
        model = Ridge(alpha=config_model['alpha'], fit_intercept=config_model['fit_intercept'], copy_X=config_model['copy_X'], max_iter=config_model['max_iter'], tol=config_model['tol'], solver=config_model['solver'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Regressor.REGRESSAO_LASSO:
        #alpha - Parâmetro de regularização (quanto maior, mais forte a regularização)
        #fit_intercept - Se True, calcula o intercepto (bias)
        #precompute - Se True, pré-computa a matriz Gram para acelerar o treinamento
        #copy_X - Se True, copia os dados de entrada
        #max_iter - Número máximo de iterações
        #tol - Tolerância para critério de parada
        #warm_start - Se True, reutiliza a solução anterior como inicialização
        #positive - Se True, força os coeficientes a serem positivos
        #random_state - Semente para reproducibilidade
        model = Lasso(alpha=config_model['alpha'], fit_intercept=config_model['fit_intercept'], precompute=config_model['precompute'], copy_X=config_model['copy_X'], max_iter=config_model['max_iter'], tol=config_model['tol'], warm_start=config_model['warm_start'], positive=config_model['positive'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Regressor.REGRESSAO_ELASTICNET:
        #alpha - Parâmetro de regularização (combina L1 e L2)
        #l1_ratio - Proporção entre L1 e L2 (0 = Ridge, 1 = Lasso)
        #fit_intercept - Se True, calcula o intercepto (bias)
        #precompute - Se True, pré-computa a matriz Gram
        #max_iter - Número máximo de iterações
        #tol - Tolerância para critério de parada
        #warm_start - Se True, reutiliza a solução anterior como inicialização
        #positive - Se True, força os coeficientes a serem positivos
        #random_state - Semente para reproducibilidade
        model = ElasticNet(alpha=config_model['alpha'], l1_ratio=config_model['l1_ratio'], fit_intercept=config_model['fit_intercept'], precompute=config_model['precompute'], max_iter=config_model['max_iter'], tol=config_model['tol'], warm_start=config_model['warm_start'], positive=config_model['positive'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Regressor.SVM:
        #kernel - Tipo de kernel: 'linear', 'poly', 'rbf', 'sigmoid'
        #degree - Grau do kernel polinomial (apenas para kernel 'poly')
        #gamma - Coeficiente do kernel (apenas para kernels 'rbf', 'poly', 'sigmoid')
        #C - Parâmetro de regularização (quanto maior, menos regularização)
        #epsilon - Margem de erro para a regressão
        #shrinking - Se True, usa a heurística de shrinking para acelerar o treinamento
        #tol - Tolerância para critério de parada
        #max_iter - Número máximo de iterações (-1 = ilimitado)
        model = SVR(kernel=config_model['kernel'], degree=config_model['degree'], gamma=config_model['gamma'], C=config_model['C'], epsilon=config_model['epsilon'], shrinking=config_model['shrinking'], tol=config_model['tol'], max_iter=config_model['max_iter'])
    elif str(config_model_job['algorithm']).lower() == Regressor.GRADIENT_BOOST:
        #n_estimators - Número de estágios de boosting
        #learning_rate - Taxa de aprendizado
        #loss - Função de perda: 'squared_error', 'absolute_error', 'huber', 'quantile'
        #max_depth - Profundidade máxima de cada árvore
        #min_samples_split - Número mínimo de amostras para dividir um nó
        #min_samples_leaf - Número mínimo de amostras em uma folha
        #random_state - Semente para reproducibilidade
        model = GradientBoostingRegressor(n_estimators=config_model['n_estimators'], learning_rate=config_model['learning_rate'], loss=config_model['loss'], max_depth=config_model['max_depth'], min_samples_split=config_model['min_samples_split'], min_samples_leaf=config_model['min_samples_leaf'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Regressor.ADA_BOOST:
        #base_estimator - Estimador base (se None, usa DecisionTreeRegressor com max_depth=3)
        #n_estimators - Número de estimadores
        #learning_rate - Taxa de aprendizado
        #loss - Função de perda: 'linear', 'square', 'exponential'
        #random_state - Semente para reproducibilidade
        model = AdaBoostRegressor(base_estimator=config_model['base_estimator'], n_estimators=config_model['n_estimators'], learning_rate=config_model['learning_rate'], loss=config_model['loss'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Regressor.ARVORE_DECISAO:
        #criterion - Função para medir a qualidade de uma divisão: 'squared_error', 'friedman_mse', 'absolute_error'
        #splitter - Estratégia para escolher a divisão: 'best', 'random'
        #max_depth - Profundidade máxima da árvore
        #min_samples_split - Número mínimo de amostras para dividir um nó
        #min_samples_leaf - Número mínimo de amostras em uma folha
        #max_features - Número máximo de features consideradas para divisão
        #random_state - Semente para reproducibilidade
        model = DecisionTreeRegressor(criterion=config_model['criterion'], splitter=config_model['splitter'], max_depth=config_model['max_depth'], min_samples_split=config_model['min_samples_split'], min_samples_leaf=config_model['min_samples_leaf'], max_features=config_model['max_features'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Regressor.FLORESTA_ALEATORIA:
        #n_estimators - Número de árvores na floresta
        #criterion - Função para medir a qualidade de uma divisão
        #max_depth - Profundidade máxima de cada árvore
        #min_samples_split - Número mínimo de amostras para dividir um nó
        #min_samples_leaf - Número mínimo de amostras em uma folha
        #max_features - Número máximo de features consideradas para divisão
        #bootstrap - Se True, usa amostragem com reposição
        #random_state - Semente para reproducibilidade
        model = RandomForestRegressor(n_estimators=config_model['n_estimators'], criterion=config_model['criterion'], max_depth=config_model['max_depth'], min_samples_split=config_model['min_samples_split'], min_samples_leaf=config_model['min_samples_leaf'], max_features=config_model['max_features'], bootstrap=config_model['bootstrap'], random_state=config_model['random_state'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def calculate_clas(pipeline, y_test, X_test, dados_validation, nomes_colunas):
    # Previsões
    X_test = pl.DataFrame(X_test, schema=nomes_colunas)
    pipeline = pipeline.named_steps['model']
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

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

    filename = f"SUPERVISED_CLASSFICATION_REF{datetime.now().strftime("%Y%m%d")}"
    FileWrite().save_file(filename=filename, dados={"acuracia":accuracy, "precisao":precision, "recall":recall, "f1":f1, "matriz_conf":conf_matrix}, type={"aprendizado": "supervised", "mode_aprendizado": "classificacao"}, validations=dados_validation)

def calculate_reg(pipeline, y_test, X_test, dados_validation, nomes_colunas):
    # Previsões
    X_test = pl.DataFrame(X_test, schema=nomes_colunas)
    pipeline = pipeline.named_steps['model']
    y_pred = pipeline.predict(X_test)

    # Cálculo das métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Erro Quadrático Médio (MSE): {mse}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    print(f"Coeficiente de Determinação (R²): {r2}")

    filename = f"SUPERVISED_REGRESSION_REF{datetime.now().strftime("%Y%m%d")}"
    FileWrite().save_file(filename=filename,
                        dados={"mse": mse, "mae": mae, "r2": r2},
                        type={"aprendizado": "supervised", "mode_aprendizado": "regressao"},
                        validations=dados_validation)