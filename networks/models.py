import networks.algorithms as algorithms
from networks import Mode

def choice_model(config_model):

    model = None
    if str(config_model['mode']['name_mode']).lower() == Mode.CLASSIFICACAO:
        print("Chamando tipo de modelo de Classificação")
        model = algorithms.get_algorithm_clas(config_model['mode'])
    elif str(config_model['mode']['name_mode']).lower() == Mode.REGRESSAO:
        print("Chamando tipo de modelo de Regressão")
        model = algorithms.get_algorithm_reg(config_model['mode'])
    elif str(config_model['mode']['name_mode']).lower() == Mode.CLUSTERING:
        print("Chamando tipo de modelo de Clusterização")
        model = algorithms.get_algorithm_clus(config_model['mode'])
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()

    return model

def statics(config_model, pipeline, y_test, X_test, y_true, X_original, labels, dados_validation, nomes_colunas):

    if str(config_model['mode']['name_mode']).lower() == Mode.CLASSIFICACAO:
        print("Calculando métricas para Classificação")
        algorithms.calculate_clas(pipeline, y_test, X_test, dados_validation, config_model, nomes_colunas)
    elif str(config_model['mode']['name_mode']).lower() == Mode.REGRESSAO:
        print("Calculando métricas para Regressão")
        algorithms.calculate_reg(pipeline, y_test, X_test, dados_validation, config_model)
    elif str(config_model['mode']['name_mode']).lower() == Mode.CLUSTERING:
        print("Calculando métricas para Clusterização")
        algorithms.calculate_clus(pipeline, X_test, y_true=y_true, labels=labels, X=X_original, dados_validation=dados_validation)
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()