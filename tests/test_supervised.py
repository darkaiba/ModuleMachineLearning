import unittest
from unittest.mock import patch
import polars as pl
from supervised import supervised as supervised
from src.getdatas import DataReaderFile

class TestSupervised(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Chama o construtor da classe pai
        self.json_supervised_classificacao = {
            "input_file": True,
            "learning": "supervised",
            "reading": {
                "reading_mode": 'json',
                "host": None,
                "user": None,
                "password": None,
                "caminho": 'C:\\Users\\ph_li\\PycharmProjects\\qfw_machine_learning\\.venv\\datasets',
                "nome_arquivo": 'iris.json',
                "database": None,
                "type_database": None,
                "query": None
            },
            "pipeline": {
                "numeric_features": ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                "categorical_features": None,
                "normalize_method_num": 'standard',
                "normalize_method_cat": None
            },
            "validations": {
                "cross_validation": True,
                "random": True,
                "grid": True,
                "param": {
                    "min_samples_split": [2, 4],
                    "min_samples_leaf": [1, 3]
                },
                "folders": 5,
                "scoring": 'accuracy'
            },
            "mode": {
                "name_mode": "classificacao",
                "algorithm": "arvore_decisao",
                "target": "target",
                "learning_network": None,
                "params": {
                    "criterion": 'entropy',
                    "max_depth": 3,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42
                }
            }
        }

        self.json_supervised_regressao = {
            "input_file": True,
            "learning": "supervised",
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
                    "min_samples_split" : [2, 4],
                    "min_samples_leaf" : [1, 3]
                },
                "folders":5,
                "scoring":'accuracy'
            },
            "mode":{
                "name_mode": "regressao",
                "algorithm": "florestas_aleatorias",
                "target": "target",
                "learning_network": None,
                "params": {
                    'n_estimators': 150,        # 150 árvores na floresta
                    'criterion': 'squared_error', # Critério de erro quadrático médio
                    'max_depth': 12,            # Profundidade máxima de 12 níveis
                    'min_samples_split': 3,     # Mínimo de 3 amostras para dividir um nó
                    'min_samples_leaf': 1,      # Mínimo de 1 amostra em uma folha
                    'max_features': 'sqrt',     # Raiz quadrada do número de features
                    'bootstrap': True,          # Usa amostragem com reposição
                    'random_state': 42          # Semente para reprodutibilidade
                }
            }
        }

    @patch('builtins.print')  # Captura a função print
    def test_supervised_classificacao(self, mock_print):
        supervised.train_model(DataReaderFile(self.json_supervised_classificacao), self.json_supervised_classificacao)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

    @patch('builtins.print')  # Captura a função print
    def test_supervised_regressao(self, mock_print):
        supervised.train_model(DataReaderFile(self.json_supervised_regressao), self.json_supervised_regressao)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

if __name__ == '__main__':
    unittest.main()