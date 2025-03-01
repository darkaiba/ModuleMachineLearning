from sklearn.preprocessing import StandardScaler, MinMaxScaler #Normalização/Padronização
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder #Codificação de Categorias
from sklearn.base import TransformerMixin

class NormalizeNum(TransformerMixin):
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = StandardScaler() if str(method).lower() == 'standard' or str(method).lower() is None else MinMaxScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler.fit_transform(X)

class NormalizeCat(TransformerMixin):
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = OneHotEncoder(handle_unknown='ignore') if str(method).lower() == 'standard' or str(method).lower() is None else LabelEncoder()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler.fit_transform(X)
