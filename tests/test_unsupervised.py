import unittest
from unittest.mock import patch
import polars as pl
from unsupervised import unsupervised as unsupervised
from src.getdatas import DataReaderFile

class TestUnsupervised(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Chama o construtor da classe pai
        self.json_unsupervised_clustering = {
            "input_file": True,
            "learning": "unsupervised",
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
                    'n_clusters': [2, 5, 8]
                },
                "folders":5,
                "scoring":'accuracy'
            },
            "mode":{
                "name_mode": "cluster",
                "algorithm": "kmeans",
                "target": None,
                "learning_network": None,
                "params": {
                    'n_clusters': 3,        # 3 clusters
                    'init': 'k-means++',    # Inicialização k-means++
                    'n_init': 10,           # 10 inicializações
                    'max_iter': 300,        # 300 iterações máximas
                    'random_state': 42      # Semente para reprodutibilidade
                }
            }
        }

        self.json_unsupervised_reduce = {
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

    @patch('builtins.print')  # Captura a função print
    def test_unsupervised_clustering(self, mock_print):
        unsupervised.train_model(DataReaderFile(self.json_unsupervised_clustering), self.json_unsupervised_clustering)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

    @patch('builtins.print')  # Captura a função print
    def test_unsupervised_reduce(self, mock_print):
        unsupervised.train_model(DataReaderFile(self.json_unsupervised_reduce), self.json_unsupervised_reduce)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

if __name__ == '__main__':
    unittest.main()