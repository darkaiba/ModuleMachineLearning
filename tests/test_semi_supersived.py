import unittest
from unittest.mock import patch
import polars as pl
from semisupervised import semisupervised as semisupervised
from src.getdatas import DataReaderFile

class TestSemiSupervised(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Chama o construtor da classe pai
        self.json_semisupervised_classificacao = {
            "input_file": True,
            "learning": "semisupervised",
            "reading":{
                "reading_mode":'json',
                "host": None,
                "user": None,
                "password": None,
                "caminho": 'C:\\Documents',
                "nome_arquivo": 'iris.json',
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
                    "kernel" : ['rbf'],
                    "n_neighbors" : [5, 7, 9]
                },
                "folders":5,
                "scoring":'accuracy'
            },
            "mode":{
                "name_mode": "classificacao",
                "algorithm": "label_propagation",
                "target": "target",
                "learning_network": None,
                "params": {
                    'kernel': 'rbf',       # Kernel Radial Basis Function (RBF)
                    'n_neighbors': 7,      # 7 vizinhos mais próximos
                    'max_iter': 1000,      # Máximo de 1000 iterações
                    'tol': 1e-3,           # Tolerância de convergência de 0.001
                    'n_jobs': -1           # Usar todos os núcleos da CPU
                }
            }
        }

    @patch('builtins.print')  # Captura a função print
    def test_semisupervised_classificacao(self, mock_print):
        semisupervised.train_model(DataReaderFile(self.json_semisupervised_classificacao), self.json_semisupervised_classificacao)
        mock_print.assert_called_with('Modelo salvo com sucesso!')  # Verifica se a função print foi chamada corretamente

if __name__ == '__main__':
    unittest.main()
