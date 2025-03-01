import unsupervised.algorithms as algorithm
from unsupervised import Mode

def choice_model(config_model):

    model = None
    if str(config_model['mode']['name_mode']).lower() == Mode.CLUSTER:
        print("Chamando tipo de modelo de Clusterização")
        model = algorithm.get_algorithm_clus(config_model['mode'])
    elif str(config_model['mode']['name_mode']).lower() == Mode.REDUCE:
        print("Chamando tipo de modelo de Redução de Dimensionalidade")
        model = algorithm.get_algorithm_red(config_model['mode'])
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()

    return model

def statics(config_model, y_test, X_test, X_original, labels, pipeline, dados_validation, nomes_colunas):

    if str(config_model['mode']['name_mode']).lower() == Mode.CLUSTER:
        print("Calculando métricas para Clusterização")
        algorithm.calculate_clus(X_test, y_true=y_test, labels=labels, dados_validation=dados_validation)
    elif str(config_model['mode']['name_mode']).lower() == Mode.REDUCE:
        print("Calculando métricas para Redução de Dimensionalidade")
        algorithm.calculate_red(pipeline, X_original, nomes_colunas, labels=labels, dados_validation=dados_validation)
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()