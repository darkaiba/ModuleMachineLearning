import semisupervised.algorithms as algorithms
from supervised import Mode

def choice_model(config_model):

    model = None
    if str(config_model['mode']['name_mode']).lower() == Mode.CLASSIFICACAO:
        print("Chamando tipo de modelo de Classificação")
        model = algorithms.get_algorithm_clas(config_model['mode'])
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()

    return model

def statics(config_model, pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas):

    if str(config_model['mode']['name_mode']).lower() == Mode.CLASSIFICACAO:
        print("Calculando métricas para Classificação")
        algorithms.calculate_clas(pipeline, y_test, X_test, random_unlabeled_points, dados_validation, nomes_colunas)
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()