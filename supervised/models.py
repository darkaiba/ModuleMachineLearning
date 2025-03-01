import supervised.algorithms as algoritm
from supervised import Mode

def choice_model(config_model):

    model = None
    if str(config_model['mode']['name_mode']).lower() == Mode.CLASSIFICACAO:
        print("Chamando tipo de modelo de Classificação")
        model = algoritm.get_algorithm_clas(config_model['mode'])
    elif str(config_model['mode']['name_mode']).lower() == Mode.REGRESSAO:
        print("Chamando tipo de modelo de Regressão")
        model = algoritm.get_algorithm_reg(config_model['mode'])
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()

    return model

def statics(config_model, pipeline, y_test, X_test, dados_validation, nomes_colunas):

    if str(config_model['mode']['name_mode']).lower() == Mode.CLASSIFICACAO:
        print("Calculando métricas para Classificação")
        algoritm.calculate_clas(pipeline, y_test, X_test, dados_validation, nomes_colunas)
    elif str(config_model['mode']['name_mode']).lower() == Mode.REGRESSAO:
        print("Calculando métricas para Regressão")
        algoritm.calculate_reg(pipeline, y_test, X_test, dados_validation, nomes_colunas)
    else:
        raise ValueError(f"Modelo não reconhecido: {config_model['mode']['name_mode']}")
        exit()