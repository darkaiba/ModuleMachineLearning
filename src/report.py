from src import Aprendizado
from supervised import Mode as mode_sup
from unsupervised import Mode as mode_unsup
from networks import Mode as mode_net

class FileWrite:

    def __init__(self):
        return

    def save_file(self, filename, dados, type, validations=None):
        with open(f"REPORT_{filename}.md", "w", encoding="utf-8") as arquivo:
            arquivo.write("<h1>Relatório de Avaliação do Modelo</h1>\n")
            arquivo.write("================================\n")

            if (type['aprendizado'] == Aprendizado.SUPERVISED or type['aprendizado'] == Aprendizado.SEMISUPERVISED) and mode_sup.CLASSIFICACAO == type['mode_aprendizado']:
                arquivo.write(f"<p>Acurácia: {dados['acuracia']}</p>\n")
                arquivo.write(f"<p>Precisão: {dados['precisao']}</p>\n")
                arquivo.write(f"<p>Recall: {dados['recall']}</p>\n")
                arquivo.write(f"<p>F1-Score: {dados['f1']}</p>\n")
                arquivo.write(f"<p>Matriz de Confusão: {dados['matriz_conf']}</p>\n")

                arquivo.write("================================\n")
                if dados['acuracia'] >= 1.0:
                    arquivo.write(f"<h3>Alerta</h3>\n")
                    arquivo.write(f"<p><b> Possivel Overfitting, sua acurácia foi de: {dados['acuracia']*100} %</b></p>\n")

            if type['aprendizado'] == Aprendizado.SUPERVISED and mode_sup.REGRESSAO == type['mode_aprendizado']:
                arquivo.write(f"<p>Erro Quadrático Médio (MSE): {dados['mse']}</p>\n")
                arquivo.write(f"<p>Erro Absoluto Médio (MAE): {dados['mae']}</p>\n")
                arquivo.write(f"<p>Coeficiente de Determinação (R²): {dados['r2']}</p>\n")

            if type['aprendizado'] == Aprendizado.UNSUPERVISED and mode_unsup.CLUSTER == type['mode_aprendizado']:
                arquivo.write("<h5>Métricas intrínsecas</h5>\n")
                arquivo.write(f"<p>Silhouette Score: {dados['silhouette_score']}</p>\n")
                arquivo.write(f"<p>Calinski-Harabasz Index: {dados['calinski_harabasz_score']}</p>\n")
                arquivo.write(f"<p>Davies-Bouldin Index: {dados['davies_bouldin_score']}</p>\n")

                arquivo.write("================================\n")
                arquivo.write("<h5>Métricas extrínsecas</h5>\n")
                arquivo.write(f"<p>Adjusted Rand Index: {dados['adjusted_rand_score']}</p>\n")
                arquivo.write(f"<p>Normalized Mutual Information: {dados['normalized_mutual_info']}</p>\n")

                arquivo.write("================================\n")
                arquivo.write("<h5>ANOVA</h5>\n")
                arquivo.write(f"<p>Resultados da ANOVA: {dados['anova']}</p>\n")

                if type['mode_aprendizado_network']:
                    arquivo.write("================================\n")
                    arquivo.write("<h5>RESULTADOS DO MAPEAMENTO</h5>\n")
                    arquivo.write(f"<img src='{type['mapa']}' alt='Mapas de Kohonem'>\n")

            if type['aprendizado'] == Aprendizado.UNSUPERVISED and mode_unsup.REDUCE == type['mode_aprendizado']:
                arquivo.write(f"<p>Reconstruction Error: {dados['reconstruction_error']}</p>\n")
                arquivo.write(f"<p>Trustworthiness: {dados['trust']}</p>\n")

            if validations is not None:
                arquivo.write("================================\n")
                arquivo.write("<h3>Validações</h3>\n")
                for validation in validations:
                    arquivo.write(f"{validation}")