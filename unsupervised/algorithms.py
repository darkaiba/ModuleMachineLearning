from sklearn.cluster import KMeans #Kmeans
from sklearn.cluster import AgglomerativeClustering #Agrupamento Hierarquico
from sklearn.cluster import DBSCAN #DBSCAN
from sklearn.mixture import GaussianMixture #GMM
from sklearn.cluster import SpectralClustering #Spectral Clustering

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import polars as pl
import numpy as np
from scipy.stats import f_oneway
from unsupervised import Agrupamento
from datetime import datetime
from src.report import FileWrite

from sklearn.decomposition import PCA #Análise de Componentes Principais (PCA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #Análise de Discriminante Linear (LDA)
from sklearn.manifold import TSNE #t-SNE
from sklearn.manifold import Isomap #Isomap
from sklearn.decomposition import FactorAnalysis #Factor Analysis
from sklearn.decomposition import NMF #Non-negative Matrix Factorization (NMF)

from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

from unsupervised import Reduce
from datetime import datetime
from src.report import FileWrite

def get_algorithm_clus(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Agrupamento.KMEANS:
        #n_clusters = Número de clusters
        #init - Método de inicialização dos centróides
        #n_init - Número de vezes que o algoritmo será executado com diferentes sementes
        #max_iter - Número máximo de iterações
        #random_state - Semente para reproducibilidade
        model = KMeans(n_clusters=config_model['n_clusters'], init=config_model['init'], n_init=config_model['n_init'], max_iter=config_model['max_iter'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Agrupamento.AGRUPAMENTO_HIERARQUICO:
        #n_clusters - Número de clusters
        #affinity - Métrica de distância (ex: 'euclidean', 'manhattan', 'cosine')
        #linkage- Critério de ligação (ex: 'ward', 'complete', 'average', 'single')
        model = AgglomerativeClustering(n_clusters=config_model['n_clusters'], affinity=config_model['affinity'], linkage=config_model['linkage'])
    elif str(config_model_job['algorithm']).lower() == Agrupamento.DBSCAN:
        #eps - Distância máxima entre dois pontos para serem considerados vizinhos
        #min_samples - Número mínimo de pontos para formar um cluster
        #metric - Métrica de distância (ex: 'euclidean', 'manhattan', 'cosine')
        model = DBSCAN(eps=config_model['eps'], min_samples=config_model['min_samples'], metric=config_model['metric'])
    elif str(config_model_job['algorithm']).lower() == Agrupamento.GMM:
        #n_components - Número de componentes (clusters)
        #covariance_type - Tipo de matriz de covariância (ex: 'full', 'tied', 'diag', 'spherical')
        #random_state - Semente para reproducibilidade
        model = GaussianMixture(n_components=config_model['n_components'], covariance_type=config_model['covariance_type'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Agrupamento.SPECTRAL:
        #n_clusters - Número de clusters
        #affinity - Método de construção da matriz de afinidade (ex: 'rbf', 'nearest_neighbors')
        #random_state - Semente para reproducibilidade
        model = SpectralClustering(n_clusters=config_model['n_clusters'], affinity=config_model['affinity'], random_state=config_model['random_state'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def get_algorithm_red(config_model_job):
    model = None
    config_model = config_model_job['params']

    if str(config_model_job['algorithm']).lower() == Reduce.PCA:
        #n_components - Número de componentes principais a serem retidos
        #whiten - Se True, normaliza os componentes principais
        #random_state - Semente para reproducibilidade
        model = PCA(n_components=config_model['n_components'], whiten=config_model['whiten'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Reduce.LDA:
        #n_components - Número de componentes a serem retidos
        model = LinearDiscriminantAnalysis(n_components=config_model['n_components'])
    elif str(config_model_job['algorithm']).lower() == Reduce.TSNE:
        #n_components - Número de dimensões no espaço reduzido
        #perplexity - Número de vizinhos considerados
        #learning_rate - Taxa de aprendizado
        #random_state - Semente para reproducibilidade
        model = TSNE(n_components=config_model['n_components'], perplexity=config_model['perplexity'], learning_rate=config_model['learning_rate'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Reduce.ISOMAP:
        #n_components - Número de dimensões no espaço reduzido
        #n_neighbors - Número de vizinhos para construir o grafo
        model = Isomap(n_components=config_model['n_components'], n_neighbors=config_model['n_neighbors'])
    elif str(config_model_job['algorithm']).lower() == Reduce.FACTOR_ANALYSIS:
        #n_components - Número de fatores latentes
        #random_state - Semente para reproducibilidade
        model = FactorAnalysis(n_components=config_model['n_components'], random_state=config_model['random_state'])
    elif str(config_model_job['algorithm']).lower() == Reduce.NMF:
        #n_components - Número de componentes
        #init - Método de inicialização
        #random_state - Semente para reproducibilidade
        model = NMF(n_components=config_model['n_components'], init=config_model['init'], random_state=config_model['random_state'])
    else:
        raise ValueError(f"Tipo de algoritmo não reconhecido: {str(config_model_job['algorithm']).lower()}")
        exit()

    return model

def calculate_clus(X_test, y_true=None, labels=None, dados_validation=None):
    silhouette = None
    calinski_harabasz = None
    davies_bouldin = None
    adjusted_rand = None
    normalized_mutual_info = None
    anova_resuts = None

    if labels is not None:
        # Métricas intrínsecas
        silhouette = silhouette_score(X_test, labels)
        calinski_harabasz = calinski_harabasz_score(X_test, labels)
        davies_bouldin = davies_bouldin_score(X_test, labels)
        print("Silhouette Score:", silhouette)
        print("Calinski-Harabasz Index:", calinski_harabasz)
        print("Davies-Bouldin Index:", davies_bouldin)
    else:
        silhouette = "Não foi possivel calcular o Score de Silhouette, pois faltam passar os rotulos"
        calinski_harabasz = "Não foi possivel calcular o Indice de Calinski-Harabasz, pois faltam passar os rotulos"
        davies_bouldin = "Não foi possivel calcular o Indice de Davies-Bouldin, pois faltam passar os rotulos"
        print(silhouette)
        print(calinski_harabasz)
        print(davies_bouldin)

    if labels is not None and y_true is not None:
        # Métricas extrínsecas (se y_true estiver disponível)
        if len(y_true) == len(labels):
            adjusted_rand = adjusted_rand_score(y_true, labels)
            normalized_mutual_info = normalized_mutual_info_score(y_true, labels)
        else:
            adjusted_rand = f"As listas que estão sendo comparadas, tem tamanhos diferentes ({y_true.shape, labels.shape})"
            normalized_mutual_info = f"As listas que estão sendo comparadas, tem tamanhos diferentes ({y_true.shape, labels.shape})"

        print("Adjusted Rand Index:", adjusted_rand)
        print("Normalized Mutual Information:", normalized_mutual_info)
    else:
        adjusted_rand = "Não foi possivel calcular o Indice de Adjusted Rand, pois faltam passar os rotulos obtidos e/ou os rótulos verdadeiros"
        normalized_mutual_info = "Não foi possivel calcular o Normalized Mutual Information (NMI), pois faltam passar os rotulos obtidos e/ou os rótulos verdadeiros"
        print(adjusted_rand)
        print(normalized_mutual_info)

    #Calculo de ANOVA
    if X_test is not None and labels is not None:
        if len(X_test) == len(labels):
            anova_results = calculate_anova(X_test, labels)
        else:
            anova_results = f"As listas que estão sendo comparadas, tem tamanhos diferentes ({X.shape, labels.shape})"
        print(f"Resultados da ANOVA: {anova_results}")
    else:
        anova_results = "Não foi possivel calcular o ANOVA, pois faltam passar os rotulos obtidos e/ou dados originais"
        print(anova_results)

    filename = f"UNSUPERVISED_CLUSTERING_REF{datetime.now().strftime("%Y%m%d")}"
    FileWrite().save_file(filename=filename,
                        dados={"silhouette_score": silhouette, "calinski_harabasz_score": calinski_harabasz, "davies_bouldin_score": davies_bouldin, "adjusted_rand_score":adjusted_rand, "normalized_mutual_info": normalized_mutual_info, "anova":anova_results},
                        type={"aprendizado": "unsupervised", "mode_aprendizado": "cluster", "mode_aprendizado_network":False},
                        validations=dados_validation)

def calculate_red(pipeline, X_original, nomes_colunas, labels=None, dados_validation=None):
    trust = None
    reconstruction_error = None
    X_original = pl.DataFrame(X_original, schema=nomes_colunas)
    X_test_transform = pipeline.fit_transform(X_original)

    # Verifica se o modelo
    model = pipeline.named_steps['model']
    if isinstance(model, LinearDiscriminantAnalysis):
        # LDA não possui inverse_transform, então não calculamos reconstruction_error
        X_test_transform = pipeline.fit_transform(X_original, labels)  # LDA requer rótulos (labels)
        reconstruction_error = "LDA não suporta cálculo de erro de reconstrução. Pegue os dados transformado e execute em outro modelo para obter as métricas."
        trust = "LDA não suporta cálculo de trustworthiness. Pegue os dados transformado e execute em outro modelo para obter as métricas."
    elif isinstance(model, TSNE):
        # t-SNE não possui inverse_transform, então não calculamos reconstruction_error
        X_test_transform = pipeline.fit_transform(X_original)
        reconstruction_error = "t-SNE não suporta cálculo de erro de reconstrução."
        trust = trustworthiness(X_original, X_test_transform)  # t-SNE suporta trustworthiness
    elif isinstance(model, Isomap):
        X_test_transform = pipeline.fit_transform(X_original)
        reconstruction_error = "Isomap não suporta cálculo de erro de reconstrução. Pegue os dados transformado e execute em outro modelo para obter as métricas."
        trust = "Isomap não suporta cálculo de trustworthiness. Pegue os dados transformado e execute em outro modelo para obter as métricas."
    else:
        if X_original is not None:
            # Erro de reconstrução (para PCA, NMF, etc.)
            X_reconstructed = model.inverse_transform(X_test_transform)
            reconstruction_error = mean_squared_error(X_original, X_reconstructed)
            print("Reconstruction Error:", reconstruction_error)

            # Trustworthiness (para t-SNE, Isomap, etc.)
            trust = trustworthiness(X_original, X_test_transform)
            print("Trustworthiness:", trust)
        else:
            reconstruction_error = "Não foi possivel calcular o Erro quadrático, pois faltam passar os dados originais"
            trust = "Não foi possivel calcular o Trustworthiness, pois faltam passar os dados originais"
            print(reconstruction_error)
            print(trust)

    filename = f"UNSUPERVISED_REDUCE_REF{datetime.now().strftime("%Y%m%d")}"
    FileWrite().save_file(filename=filename,
                        dados={"trust": trust,"reconstruction_error": reconstruction_error},
                        type={"aprendizado": "unsupervised", "mode_aprendizado": "reduce", "mode_aprendizado_network":False},
                        validations=dados_validation)

def calculate_anova(X, labels):
    """
    Calcula a ANOVA para cada feature em relação aos clusters.

    :param X: Dados de entrada (numpy array ou polars DataFrame).
    :param labels: Rótulos dos clusters (numpy array).
    :return: DataFrame do polars com os resultados da ANOVA (F-statistic e p-value) para cada feature.
    """
    # Converte X para polars DataFrame (se ainda não for)
    if not isinstance(X, pl.DataFrame):
        X = pl.DataFrame(X)

    # Listas para armazenar os resultados
    features = X.columns
    f_stats = []
    p_values = []

    # Calcula a ANOVA para cada feature
    for feature in features:
        # Extrai os valores da feature e agrupa por cluster
        feature_values = X[feature].to_numpy()
        groups = [feature_values[labels == cluster] for cluster in np.unique(labels)]

        # Calcula a ANOVA
        f_stat, p_value = f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_value)

    # Cria um DataFrame do polars com os resultados
    anova_results = pl.DataFrame({
        'Feature': features,
        'F-statistic': f_stats,
        'p-value': p_values
    })

    return anova_results


