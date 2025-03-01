from sklearn.semi_supervised import LabelPropagation #Label Propagation
from sklearn.semi_supervised import LabelSpreading #Label Spreading
from semisupervised import Classificador

from datetime import datetime
from src.report import FileWrite

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

import polars as pl

def get_algorithm_clas(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Classificador.LABEL_PROPAGATION:
        #kernel - Tipo de kernel usado para calcular a similaridade entre as amostras
        #n_neighbors - Numero de vizinhos
        #max_iter - Número máximo de iterações
        #tol - Tolerância para convergência
        #n_jobs -  Número de jobs paralelos para executar o algoritmo
        model = LabelPropagation(kernel=config_model['kernel'], n_neighbors=config_model['n_neighbors'], max_iter=config_model['max_iter'], tol=config_model['tol'], n_jobs=config_model['n_jobs'])
    elif str(config_model_job['algorithm']).lower() == Classificador.LABEL_SPREADING:
        # kernel - Tipo de kernel usado para calcular a similaridade entre as amostras
        # n_neighbors - Numero de vizinhos
        #alpha - Fator de suavização
        #max_iter - Número máximo de iterações
        #tol - Tolerância para convergência
        #n_jobs - Número de jobs paralelos para executar o algoritmo
        model = LabelSpreading(kernel=config_model['kernel'], n_neighbors=config_model['n_neighbors'], alpha=config_model['alpha'], max_iter=config_model['max_iter'], tol=config_model['tol'], n_jobs=config_model['n_jobs'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def calculate_clas(pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas):
    # Previsões
    X_test = pl.DataFrame(X_test, schema=nomes_colunas)
    pipeline = pipeline.named_steps['model']
    y_pred_unlabeled = pipeline.predict(X_test)  # Rótulos inferidos para os dados não rotulados

    # Cálculo das métricas
    accuracy = accuracy_score(y_test, y_pred_unlabeled)
    precision = precision_score(y_test, y_pred_unlabeled, average='macro')
    recall = recall_score(y_test, y_pred_unlabeled, average='macro')
    f1 = f1_score(y_test, y_pred_unlabeled, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred_unlabeled)

    print(f"Accuracia: {accuracy}")
    print(f"Precisão: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Matriz de Confusão:\n{conf_matrix}")

    if accuracy >= 1.0:
        print("Possivel Overfitting")

    filename = f"SEMISUPERVISED_CLASSFICATION_REF{datetime.now().strftime("%Y%m%d")}"
    FileWrite().save_file(filename=filename,
                        dados={"acuracia": accuracy, "precisao": precision, "recall": recall, "f1": f1,
                               "matriz_conf": conf_matrix},
                        type={"aprendizado": "semisupervised", "mode_aprendizado": "classificacao"},
                        validations=dados_validation)
