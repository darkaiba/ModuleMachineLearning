import supervised.supervised as supervised
import unsupervised.unsupervised as unsupervised
import semisupervised.semisupervised as semisupervised
import networks.network as networks
from src.getdatas import DataReaderRemote, DataReaderFile
from src import Aprendizado
from src.configJson import EvaluateJson

def main(job):

    reader = None
    if job['input_file'] is None or job['input_file'] is False:
        # Busca dados em um servidor remoto ou banco de dados
        print("Buscando os dados")
        reader = DataReaderRemote(job)
    else:
        # Faz a leitura do arquivo de entrada (Local)
        print("Lendo o Arquivo de Entrada")
        reader = DataReaderFile(job)

    if str(job['learning']).lower() == Aprendizado.SUPERVISED:
        print("Chamando o Aprendizado Supervisionado")
        supervised.train_model(reader, job)
    elif str(job['learning']).lower() == Aprendizado.UNSUPERVISED:
        print("Chamando o Aprendizado Não Supervisionado")
        unsupervised.train_model(reader, job)
    elif str(job['learning']).lower() == Aprendizado.SEMISUPERVISED:
        print("Chamando o Aprendizado Semi Supervisionado")
        semisupervised.train_model(reader, job)
    elif str(job['learning']).lower() == Aprendizado.NETWORK:
        print("Chamando o Aprendizado por Redes")
        networks.train_model(reader, job)
    elif str(job['learning']).lower() == Aprendizado.LLM:
        print("Chamando o Aprendizado por Modelo de Linguagem Grande (Large Language Model)")
    else:
        raise ValueError(f"Tipo de aprendizado não reconhecido: {str(job['learning']).lower()}")
        exit()

if __name__ == "__main__":
    print("Iniciando o processo de criação de modelo de IA")

    #Passar a porcentagem do slip de dados de treino e de validação
    json_job = {
        "input_file": True,
        "learning": "unsupervised",
        "reading":{
            "reading_mode":'csv',
            "host": None,
            "user": None,
            "password": None,
            "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
            "nome_arquivo": 'iris.csv',
            "database": None,
            "type_database": None,
            "query": None
        },
        "pipeline": {
			"numeric_features": ['sepal_length','sepal_width','petal_length','petal_width'],
			"categorical_features": None,
			"normalize_method_num":'minmax',
			"normalize_method_cat": None
		},
        "validations":{
            "cross_validation": True,
            "random": True,
            "grid": True,
            "param":{
                'n_components': [2, 5, 8]
            },
            "folders":5,
            "scoring":'accuracy'
        },
        "mode":{
            "name_mode": "reduce",
            "algorithm": "pca",
            "target": "target",
            "learning_network": None,
            "params": {
                'n_components': 2,    # 2 componentes principais
                'whiten': True,       # Normaliza os componentes principais
                'random_state': 42    # Semente para reprodutibilidade
            }
        }
    }

    print("Validando Json...")
    json_job_evaluate = EvaluateJson().evaluate(json_job)
    print("Json de entrada validado!")
    main(json_job_evaluate)