import unittest
from unittest.mock import patch
import polars as pl
from networks import network as network
from src.getdatas import DataReaderFile
from scipy.stats import randint

class TestNetwork(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Chama o construtor da classe pai
        self.json_network_classificacao = {
            "input_file": True,
            "learning": "network",
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
                    "activation" : ['relu'],
                    "learning_rate" : ['adaptive']
                },
                "folders":5,
                "scoring":'accuracy'
            },
            "mode":{
                "name_mode": "classificacao",
                "algorithm": "mlp",
                "target": "target",
                "learning_network": "supervised",
                "params": {
                    "hidden_layer_sizes" : (100,),  # Uma camada oculta com 100 neurônios
                    "activation" : 'relu',          # Função de ativação ReLU
                    "solver" : 'adam',              # Algoritmo de otimização Adam
                    "alpha" : 0.0001,               # Constante de regularização L2
                    "learning_rate" : 'adaptive',   # Taxa de aprendizado adaptativa
                    "max_iter" : 500,               # Número máximo de iterações
                    "tol" : 1e-4,                   # Tolerância para parada
                    "random_state" : 42,           # Semente para reprodutibilidade
                    'power': False
                }
            }
        }

        self.json_network_regressao = {
            "input_file": True,
            "learning": "network",
            "reading":{
                "reading_mode":'json',
                "host": None,
                "user": None,
                "password": None,
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": 'diabetes.json',
                "database": None,
                "type_database": None,
                "query": None
            },
            "pipeline": {
                    "numeric_features": ['data_0', 'data_1', 'data_2', 'data_3', 'data_4', 'data_5', 'data_6', 'data_7','data_8', 'data_9'],
                    "categorical_features": None,
                    "normalize_method_num":'standard',
                    "normalize_method_cat": None
            },
            "validations":{
                "cross_validation": True,
                "random": True,
                "grid": True,
                "param":{
                    "activation" : ['relu'],
                    "learning_rate" : ['adaptive']
                },
                "folders":5,
                "scoring":'accuracy'
            },
			"mode":{
				"name_mode": "regressao",
				"algorithm": "mlp",
				"target": "target",
				"learning_network": "supervised",
				"params": {
					"hidden_layer_sizes" : (100,),  # Uma camada oculta com 100 neurônios
					"activation" : 'relu',          # Função de ativação ReLU
					"solver" : 'adam',              # Algoritmo de otimização Adam
					"alpha" : 0.0001,               # Constante de regularização L2
					"learning_rate" : 'adaptive',   # Taxa de aprendizado adaptativa
					"max_iter" : 500,               # Número máximo de iterações
					"tol" : 1e-4,                   # Tolerância para parada
					"random_state" : 42,           # Semente para reprodutibilidade
                    "power": False
				}
			}
        }

        self.json_network_som = {
            "input_file": True,
            "learning": "network",
            "reading":{
                "reading_mode":'parquet',
                "host": None,
                "user": None,
                "password": None,
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": 'blobs.parquet',
                "database": None,
                "type_database": None,
                "query": None
            },
            "pipeline": None,
            "validations":{
                "cross_validation": True,
                "random": True,
                "grid": True,
                "param":{
                    "sigma" : [0.5, 1],
                    "learning_rate" : [0.5, 0.8]
                },
                "folders":5,
                "scoring":'accuracy'
            },
            "mode":{
                "name_mode": "cluster",
                "algorithm": "som",
                "target": None,
                "learning_network": "unsupervised",
                "params": {
                    'map_size_1': 10,       # Tamanho do mapa na dimensão 1
                    'map_size_2': 10,       # Tamanho do mapa na dimensão 2
                    'input_len': None, # Número de features (dimensões dos dados)
                    'sigma': 1.0,           # Raio inicial do vizinho
                    'learning_rate': 0.5,   # Taxa de aprendizado inicial
                    'random_seed': 42,      # Semente para reprodutibilidade
                    'epoch': 10,
                    'power': False
                }
            }
        }

        self.json_network_dnn_reg = {
            "input_file": True,
            "learning": "network",
            "reading":{
                "reading_mode":'csv',
                "host": None,
                "user": None,
                "password": None,
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": 'diabetes.csv',
                "database": None,
                "type_database": None,
                "query": None
            },
            "pipeline": {
                "numeric_features": ['data_0', 'data_1', 'data_2', 'data_3', 'data_4', 'data_5', 'data_6', 'data_7','data_8', 'data_9'],
                "categorical_features": None,
                "normalize_method_num":'standard',
                "normalize_method_cat": None
            },
            "validations":{
                "cross_validation": True,
                "random": True,
                "grid": True,
                "param":{
                    'loss': ['mse'],
                    'metrics': ['accuracy']
                },
                "folders":5,
                "scoring":'mae'
            },
            "mode":{
                "name_mode": "regressao",
                "algorithm": "dnn",
                "target": 'target',
                "learning_network": "supervised",
                "params": {
                    'n_camadas': 2,  # Número de camadas ocultas (inteiro positivo)
                    'activation_enter': 'relu',  # Função de ativação da camada de entrada (string: 'relu', 'tanh', 'linear', etc.)
                    'n_neuronio_enter': 128,  # Número de neurônios na camada de entrada (inteiro positivo)
                    'activation_intermediate': 'relu',  # Função de ativação das camadas intermediárias (string)
                    'n_neuronio_intermediate': 64,  # Número de neurônios nas camadas intermediárias (inteiro positivo)
                    'activation_end': 'linear',  # Função de ativação da camada de saída (string: 'linear' para regressão)
                    'n_neuronio_end': 1,  # Número de neurônios na camada de saída (1 para regressão)
                    'input_shape': (10,),  # Forma dos dados de entrada (tupla de inteiros)
                    'optimizer': 'adam',  # Otimizador para treinamento (string: 'adam', 'sgd', 'rmsprop', etc.)
                    'loss': 'mse',  # Função de perda (string: 'mse', 'mae', etc.)
                    'metrics': ['mae'],  # Métricas para avaliação (lista de strings - exemplo: 'mae' para erro absoluto médio)
                    'epoch': 10,
                    'batch_size': 32,
                    'power': True
                }
            }
        }

        self.json_network_dnn_clas = {
            "input_file": True,
            "learning": "network",
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
                    'optimizer': ['adam'],
                    'loss': ['sparse_categorical_crossentropy'],
                    'metrics': ['accuracy']
                },
                "folders":5,
                "scoring":'accuracy'
            },
            "mode":{
                "name_mode": "classificacao",
                "algorithm": "dnn",
                "target": 'target',
                "learning_network": "supervised",
                "params": {
                    'n_camadas': 5,  # Número de camadas intermediárias
                    'activation_enter': 'relu',
                    'n_neuronio_enter': 256,
                    'activation_intermediate': 'relu',
                    'n_neuronio_intermediate': 128,
                    'activation_end': 'sigmoid',
                    'n_neuronio_end': 3,
                    'input_shape': (4,),  # 10 features de entrada
                    'optimizer': 'adam',
                    'loss': 'sparse_categorical_crossentropy',
                    'metrics': ['accuracy'],
                    'epoch': 10,
                    'batch_size': 32,
                    'power': True
                }
            }
        }

    @patch('builtins.print')  # Captura a função print
    def test_network_classificacao(self, mock_print):
        network.train_model(DataReaderFile(self.json_network_classificacao), self.json_network_classificacao)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

    @patch('builtins.print')  # Captura a função print
    def test_network_regressao(self, mock_print):
        network.train_model(DataReaderFile(self.json_network_regressao), self.json_network_regressao)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

    @patch('builtins.print')  # Captura a função print
    def test_network_som(self, mock_print):
        network.train_model(DataReaderFile(self.json_network_som), self.json_network_som)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

    @patch('builtins.print')  # Captura a função print
    def test_network_dnn_clas(self, mock_print):
        network.train_model(DataReaderFile(self.json_network_dnn_clas), self.json_network_dnn_clas)
        mock_print.assert_called_with(
            'Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

    @patch('builtins.print')  # Captura a função print
    def test_network_dnn_reg(self, mock_print):
        network.train_model(DataReaderFile(self.json_network_dnn_reg), self.json_network_dnn_reg)
        mock_print.assert_called_with(
            'Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

if __name__ == '__main__':
    unittest.main()