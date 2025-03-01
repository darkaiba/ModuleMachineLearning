from sklearn.model_selection import cross_val_score #Validação Cruzada
from sklearn.model_selection import GridSearchCV #GRid Search
from sklearn.model_selection import RandomizedSearchCV #Random Search

from sklearn.preprocessing import StandardScaler, MinMaxScaler #Normalização/Padronização
from sklearn.preprocessing import OneHotEncoder, LabelEncoder #Codificação de Categorias
from sklearn.impute import SimpleImputer #Imputação de Dados Faltantes
from sklearn.pipeline import Pipeline #Pipeline
from sklearn.compose import ColumnTransformer

from src.normalize import NormalizeNum, NormalizeCat
from src import Aprendizado
from networks import Mode
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold

import polars as pl
import numpy as np

class ValidationDatas:

    def __init__(self):
        self.lines = []
        return

    def validar_cruzamentos(self, model, X_test, y_test, config_model, folders, scoring):
        print("Calculando a validação cruzada")
        cv_scores = None
        kf = KFold(n_splits=folders)

        if y_test is None:
            cv_scores = cross_val_score(model, X_test, cv=kf, scoring=scoring)
            self.lines.append(f"<p>Scores da Validação Cruzada: {cv_scores}</p>\n")
            self.lines.append(f"<p>Média dos Scores: {cv_scores.mean()}</p>\n")
        elif len(X_test) == len(y_test):
            if config_model['learning'] == Aprendizado.NETWORK and 'power' in config_model['mode']['params']:
                if config_model['mode']['params']['power'] is True:
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    self.lines.append(f"<p> Validação cruzada, para redes do TensowFlow: Loss: {loss}, Accuracy: {accuracy}</p>\n")
                else:
                    cv_scores = cross_val_score(model, X_test, y_test, cv=kf, scoring=scoring)
                    self.lines.append(f"<p>Scores da Validação Cruzada: {cv_scores}</p>\n")
                    self.lines.append(f"<p>Média dos Scores: {cv_scores.mean()}</p>\n")
            else:
                cv_scores = cross_val_score(model, X_test, y_test, cv=kf, scoring=scoring)
                self.lines.append(f"<p>Scores da Validação Cruzada: {cv_scores}</p>\n")
                self.lines.append(f"<p>Média dos Scores: {cv_scores.mean()}</p>\n")
        else:
            self.lines.append(f"<p>Não foi possivel calcular, pois seu modelo resultou em tamanhos diferentes: Dados de validação --> {len(X_test)}. Rótulos de validação --> {len(y_test)}.</p>\n")

    def busca_hiperparametros_grid(self, param, model, X_test, y_test, folders, scoring):
        print("Buscando por Hiperparametros Grid")
        param_grid = param # Grade de parâmetros
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=folders, scoring=scoring)
        if y_test is None:
            grid_search.fit(X_test)
        else:
            grid_search.fit(X_test, y_test)
        self.lines.append(f"<p>Melhores parâmetros (GRID): {grid_search.best_params_}</p>\n")
        self.lines.append(f"<p>Melhor score (GRID): {grid_search.best_score_}</p>\n")
        return grid_search.best_estimator_  # Retorna o melhor modelo encontrado

    def busca_hiperparametros_random(self, param, model, X_test, y_test, folders, scoring):
        print("Buscando por Hiperparametros Random")
        param_dist = param # Distribuição de parâmetros
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=folders, scoring=scoring)
        if y_test is None:
            random_search.fit(X_test)
        else:
            random_search.fit(X_test, y_test)
        self.lines.append(f"<p>Melhores parâmetros (Random): {random_search.best_params_}</p>\n")
        self.lines.append(f"<p>Melhor score (Random): {random_search.best_score_}</p>\n")
        return random_search.best_estimator_  # Retorna o melhor modelo encontrado

    def get_validations(self, config_model, pipeline, X_test, y_test):
        pipeline_learning = pipeline
        model = pipeline.named_steps['model']

        # Faz validação cruzada
        if config_model['validations']['cross_validation']:
            if config_model['learning']==Aprendizado.NETWORK  and config_model['mode']['name_mode']==Mode.CLUSTERING:
                print("Essa métrica não se aplica a rede SOM.")
            else:
                if config_model['validations']['folders'] is None or config_model['validations']['scoring'] is None:
                    raise ValueError(
                        'São necessários o numero de pastas e Scoring que você quer avaliar. Algum e/ou ambos os valores estão nulos')
                    exit()
                else:
                    self.validar_cruzamentos(model, X_test, y_test, config_model, config_model['validations']['folders'],
                                                        config_model['validations']['scoring'])

        # Busca hiperparametros do tipo grid
        if config_model['validations']['grid']:
            if config_model['learning']==Aprendizado.NETWORK  and config_model['mode']['name_mode']==Mode.CLUSTERING:
                self.lines.append("<p>A métrica GRID não se aplica a rede SOM.</p>\n")

            elif config_model['learning']==Aprendizado.NETWORK  and config_model['mode']['params']['power'] is True:
                self.lines.append("<p>A métrica GRID não se aplica a rede do TensorFlow</p>\n")

            else:
                if config_model['validations']['folders'] is None or config_model['validations']['scoring'] is None or \
                        config_model['validations']['param'] is None:
                    raise ValueError(
                        'São necessários o numero de pastas, Scoring que você quer avaliar e o parametro de distribuição para o hiperparametro (GRID). Algum e/ou ambos os valores estão nulos')
                    exit()
                else:
                    pipeline_learning.named_steps['model'] = self.busca_hiperparametros_grid(config_model['validations']['param'], model, X_test,
                                                                       y_test, config_model['validations']['folders'],
                                                                       config_model['validations']['scoring'])

        # Busca hiperparametros randomicamente
        if config_model['validations']['random']:
            if config_model['learning'] == Aprendizado.NETWORK and config_model['mode'][
                'name_mode'] == Mode.CLUSTERING:
                self.lines.append("<p>A métrica Random não se aplica a rede SOM.</p>\n")

            elif config_model['learning'] == Aprendizado.NETWORK and config_model['mode']['params']['power'] is True:
                self.lines.append("<p>A métrica Random não se aplica a rede do TensorFlow.</p>\n")

            else:
                if config_model['validations']['folders'] is None or config_model['validations']['scoring'] is None or \
                        config_model['validations']['param'] is None:
                    raise ValueError(
                        'São necessários o numero de pastas, Scoring que você quer avaliar e o parametro de distribuição para o hiperparametro (Random). Algum e/ou ambos os valores estão nulos')
                    exit()
                else:
                    pipeline_learning.named_steps['model'] = self.busca_hiperparametros_random(config_model['validations']['param'], model,
                                                                         X_test, y_test,
                                                                         config_model['validations']['folders'],
                                                                         config_model['validations']['scoring'])
        if self.lines == []:
            self.lines = None

        return pipeline_learning, self.lines

class CreatePipeline:

    def __init__(self):
        return

    def create_preprocessing_pipeline(self, numeric_features, categorical_features, normalize_method_num, normalize_method_cat):

        preprocessor = None

        # Pipeline para features numéricas
        if numeric_features is not None:
            print("Criando o pré-processamento para variaveis numéricas")
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Imputação de dados faltantes
                ('normalize', NormalizeNum(method=normalize_method_num))  # Normalização
            ])

        # Pipeline para features categóricas
        if categorical_features is not None:
            print("Criando o pré-processamento para variaveis categóricas")
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputação de dados faltantes
                ('onehot', NormalizeCat(method=normalize_method_cat))  # Normalizacao
            ])

        # Combina os pipelines numéricos e categóricos
        if numeric_features is not None and categorical_features is not None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
        elif numeric_features is not None and categorical_features is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ])
        elif numeric_features is None and categorical_features is not None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features)
                ])

        return preprocessor

    # Verifica se as colunas são strings (DataFrame do Polars) ou índices (array NumPy)
    def get_columns(self, input_features, X):
        if input_features is None:
            return None

        # Se X for um DataFrame (Pandas ou Polars), converte os nomes das colunas para índices
        if hasattr(X, 'columns'):  # Verifica se X é um DataFrame
            return [X.get_column_index(col) for col in input_features]
        # Se X for um array NumPy, assume que input_features já são índices
        elif isinstance(X, np.ndarray):
            return input_features
        else:
            raise ValueError("O input X deve ser um DataFrame do Polars ou um array NumPy.")

    def choose_model_ml(self, config_model, modeling):

        num_epochs = None
        batch_size = None

        if str(config_model['learning']).lower() == Aprendizado.NETWORK:
            if config_model['mode']['params']['power']:
                num_epochs = config_model['mode']['params']['epoch']
                batch_size = config_model['mode']['params']['batch_size']

        # Escolhendo o modelo e suas configurações
        model_learning = modeling.choice_model(config_model)

        pipeline = None
        if config_model['pipeline'] is not None:
            print("Criando pipeline customizada para o modelo")
            # Cria um pipeline com normalização
            # Define as colunas numéricas e categóricas
            numeric_features = config_model['pipeline']['numeric_features'] if config_model['pipeline'][
                                                                                   'numeric_features'] is not None else None
            categorical_features = config_model['pipeline']['categorical_features'] if config_model['pipeline'][
                                                                                           'categorical_features'] is not None else None

            # Cria o pipeline de pré-processamento
            preprocessor = self.create_preprocessing_pipeline(numeric_features=numeric_features,
                                                                          categorical_features=categorical_features,
                                                                          normalize_method_num=config_model['pipeline'][
                                                                              'normalize_method_num'],
                                                                          normalize_method_cat=config_model['pipeline'][
                                                                               'normalize_method_cat'])

            if str(config_model['learning']).lower() == Aprendizado.NETWORK:
                model = None
                if config_model['mode']['name_mode'] == Mode.CLASSIFICACAO:
                    #model = KerasClassifier(model=model_learning, epochs=num_epochs, batch_size=batch_size, verbose=0)
                    model = model_learning

                if config_model['mode']['name_mode'] == Mode.REGRESSAO:
                    #model = KerasRegressor(model=model_learning, epochs=num_epochs, batch_size=batch_size, verbose=0)
                    model = model_learning

                # Cria um pipeline completo com pré-processamento e modelo
                if config_model['mode']['params']['power']:
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                else:
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', model_learning)
                    ])
            else:
                # Cria um pipeline completo com pré-processamento e modelo
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model_learning)
                ])
        else:
            print("Criando pipeline geral para o modelo")

            if str(config_model['learning']).lower() == Aprendizado.NETWORK:
                model = None
                if config_model['mode']['name_mode'] == Mode.CLASSIFICACAO:
                    #model = KerasClassifier(model=model_learning, epochs=num_epochs, batch_size=batch_size, verbose=0)
                    model = model_learning

                if config_model['mode']['name_mode'] == Mode.REGRESSAO:
                    #model = KerasRegressor(model=model_learning, epochs=num_epochs, batch_size=batch_size, verbose=0)
                    model = model_learning

                # Cria um pipeline sem normalização
                if config_model['mode']['params']['power']:
                    pipeline = Pipeline(steps=[
                        ('model', model)
                    ])
                else:
                    pipeline = Pipeline(steps=[
                        ('model', model_learning)
                    ])
            else:
                # Cria um pipeline sem normalização
                print("Criando pipeline geral para o modelo")
                pipeline = Pipeline(steps=[
                    ('model', model_learning)
                ])

        return pipeline