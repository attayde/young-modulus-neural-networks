
########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# Análise Computacional de Arquiteturas e Regularização por Dropout em Redes Neurais para Predição do Módulo de Young de Ligas de Titânio
#
# **Computational Analysis of Neural Network Architectures and Dropout Regularization for Young's Modulus Prediction of Titanium Alloys**
#
# *Notebook associado ao trabalho a ser apresentado no EAMC/LNCC — 2027.*
# OBS.: TODO O CODE FOI EXECUTADO ORIGINALEMNTE NO GOOGLE COLABORATOY
# <font color="blue"><b>Objetivos</b></font>:
# Este estudo consiste em analisar a influência de diferentes arquiteturas de redes neurais artificiais e o efeito da regularização (com e sem Dropout), na predição do módulo de Young de ligas de titânio. O objetivo não é desenvolver novos códigos computacionais para essa finalidade, mas analisar o desempenho preditivo de diferentes arquiteturas e a capacidade de generalização (i.e.,potencial para estimar o módulo de Young para novos dados experimentais), além de evidenciar a importância da Matemática Aplicada no contexto de abordagens orientadas por dados (data driven approach) aplicadas à Engenharia.
#
# <font color="blue"><b>Abstract</b></font>: This study investigates the influence of different artificial neural network architectures and the effect of regularization, with and without Dropout, on the prediction of the Young's modulus of titanium alloys. The objective is not to develop new computational codes for this purpose, but rather to analyze the predictive performance of different architectures and their generalization capability (i.e., their potential to estimate Young's modulus for new experimental data). The study also highlights the importance of Applied Mathematics in the context of data-driven approaches applied to Engineering.

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **1. Bibliotecas**
#
# *Geração e minipulação dos dados, construção e treinamento das redes neurais, validação cruzada e cálculo das métricas de avaliação.*

########################################################################
# CÉLULA DE CÓDIGO 3
# Célula 1
# Bibliotecas utilizadas

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping


import time
from IPython.display import clear_output, display, HTML



# Reprodutibilidade
# Define uma semente fixa para o gerador de números aleatórios,
# permitindo reproduzir os mesmos resultados em diferentes execuções.

SEED = 42
np.random.seed(SEED)

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **2. Geração de dados sintéticos**
#
# <font color="blue"><b>Objetivos</b></font>:
# Para permitir a execução e reprodução do código sem disponibilizar os dados experimentais utilizados no estudo, é gerado um conjunto de dados sintético com a mesma dimensão estrutural do conjunto original.
#
# O conjunto possui 336 amostras e 31 colunas, sendo uma identificação da amostra, 29 variáveis de entrada e uma variável-alvo correspondente ao módulo de Young.
#
# Os dados sintéticos são utilizados exclusivamente para demonstrar a execução e a estrutura do código, não sendo empregados para reproduzir os resultados experimentais apresentados no estudo.
#
# **OBS.:** No dataset orginal, foram acrescidas flags, para ligar e desligar features relativas à existência de tratamentos térmicos e mecânicos das ligas. Essa etapa foi feita ainda com os dados brutos, para posteriormente termos o dataframe inicial.

########################################################################
# CÉLULA DE CÓDIGO 5
# Célula 2
# Dimensões do dataset

N_AMOSTRAS = 336
N_FEATURES = 29

# Identificação das amostras
amostras = [f"Amostra_{i+1:03d}" for i in range(N_AMOSTRAS)]

# Geração das features
X = np.random.normal(loc=0, scale=1, size=(N_AMOSTRAS, N_FEATURES))

# Geração do target sintético, com dependência aleatória de algumas variáveis de entrada
ruido = np.random.normal(loc=0, scale=5, size=N_AMOSTRAS)
y = 100 + 10*X[:, 0] + 5*X[:, 1] - 3*X[:, 2] + ruido

# Criação do DataFrame
nomes_features = [f"Feature_{i+1:02d}" for i in range(N_FEATURES)]
df = pd.DataFrame(X, columns=nomes_features)
df.insert(0, "Amostras", amostras)
df["Young_Modulus"] = y

# Verificação
print("Dimensão:", df.shape)
df.head()

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **3. Separação das variáveis**
#
# <font color="blue"><b>Objetivos</b></font>:
# O identificador das amostras é mantido separadamente para possibilitar a rastreabilidade dos dados. As demais variáveis são divididas em entradas (X) e variável-alvo (y), correspondente ao módulo de Young.

########################################################################
# CÉLULA DE CÓDIGO 7
# Célula 3
# Separação das variáveis: organização dos dados em vetores de entrada (X) e saída (y)
id_amostras = df["Amostras"]
X = df.drop(columns=["Amostras", "Young_Modulus"])
y = df["Young_Modulus"]

print("X:", X.shape)
print("y:", y.shape)

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **3.1. Verificação de valores ausentes**
#
# <font color="blue"><b>Objetivos</b></font>: É realizada uma verificação para identificar a presença de valores ausentes no conjunto de dados antes da etapa de treinamento.
#
# **OBS.:** No dataset original foram realizadas várias etapas de pré-processamento e também AED.

########################################################################
# CÉLULA DE CÓDIGO 9
# Célula 3.1
# Verificação de valores ausentes
print("Valores NaN:", df.isna().sum().sum())

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **4. Separação dos dados em desenvolvimento e teste**
#
# <font color="blue"><b>Objetivos</b></font>:
# Os dados são divididos em 80% para desenvolvimento e 20% para teste final.
#
# O conjunto de desenvolvimento é utilizado para treinamento, validação cruzada e comparação entre as arquiteturas. O conjunto de teste permanece separado e não participa dessas etapas, sendo reservado para a avaliação final do modelo selecionado.
#
# A divisão é realizada de forma aleatória, utilizando uma semente fixa (SEED = 42) para garantir a reprodutibilidade.

########################################################################
# CÉLULA DE CÓDIGO 11
# Célula 4
# Separação dos dados em desenvolvimento e teste:
# 80% para desenvolvimento do modelo e 20% para teste final.
TEST_SIZE = 0.20

X_dev, X_test, y_dev, y_test, id_dev, id_test = train_test_split(X, y, id_amostras,test_size=TEST_SIZE,random_state=SEED,shuffle=True)

print("Desenvolvimento:", X_dev.shape)
print("Teste:", X_test.shape)

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **4.1 Configuração da validação cruzada**
#
# <font color="blue"><b>Objetivos</b></font>:
# A seleção e avaliação das arquiteturas é realizada utilizando validação cruzada com 5 folds.
#
# Os dados do conjunto de desenvolvimento serão embaralhados antes da divisão, mantendo-se a mesma semente aleatória utilizada nas demais etapas.

########################################################################
# CÉLULA DE CÓDIGO 13
# Célula 4.1
# Instanciação e configuração da validação cruzada: define 5 folds
# e os parâmetros de embaralhamento e reprodutibilidade.

N_SPLITS = 5  # Número de partes (folds) em que os dados serão divididos durante a validação cruzada

kf = KFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)

print(f"Validação cruzada: {N_SPLITS} folds")

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **4.2. Hiperparâmetros**
#
# <font color="blue"><b>Objetivos</b></font>:
# São definidos os principais hiperparâmetros utilizados no treinamento das redes neurais, incluindo taxa de aprendizagem, tamanho do lote, número máximo de épocas, paciência para o Early Stopping e taxa de Dropout.
#
# A taxa de Dropout é inicialmente definida como 0,20. Esse valor pode ser alterado para 0,00 caso se deseje executar as redes sem regularização por Dropout.

########################################################################
# CÉLULA DE CÓDIGO 15
# Célula 4.2
# Hiperparâmetros
LEARNING_RATE = 0.0005
BATCH_SIZE = 16
EPOCHS = 2000
PATIENCE = 150
DROPOUT_RATE = 0.20

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **4.3. Arquiteturas**
#
# <font color="blue"><b>Objetivos</b></font>:
# São consideradas oito arquiteturas de redes neurais artificiais, com diferentes números de neurônios e camadas ocultas.
#
# Todas as arquiteturas possuem 29 variáveis de entrada e uma saída correspondente ao módulo de Young. A comparação entre as arquiteturas permite analisar a influência da complexidade da rede no desempenho preditivo.

########################################################################
# CÉLULA DE CÓDIGO 17
# Célula 4.3
# Arquiteturas das redes neurais
ARQUITETURAS = {
    "ANN_01": [16],
    "ANN_02": [32],
    "ANN_03": [64],
    "ANN_04": [32, 16],
    "ANN_05": [64, 32],
    "ANN_06": [128, 64],
    "ANN_07": [128, 64, 32],
    "ANN_08": [256, 128]
}

print("Arquiteturas configuradas:", len(ARQUITETURAS))

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **5. Construção das redes neurais**
#
# <font color="blue"><b>Objetivos</b></font>:
# A função a seguir cria as redes neurais a partir da arquitetura especificada.
#
# As camadas ocultas utilizam a função de ativação ReLU, enquanto a camada de saída possui ativação linear, adequada à predição de uma variável contínua. O treinamento utiliza o otimizador Adam e a função de perda MSE.
#
# A regularização por Dropout é aplicada de acordo com o valor definido em DROPOUT_RATE.

########################################################################
# CÉLULA DE CÓDIGO 19
# Célula 5
# Construção e compilação das redes neurais a partir da arquitetura definida,
# permitindo avaliar diferentes configurações com aplicação opcional de Dropout.

def criar_modelo(input_dim, arquitetura):
    modelo = Sequential()

    for i, unidades in enumerate(arquitetura):
        if i == 0:
            modelo.add(Dense(unidades, activation="relu", input_shape=(input_dim,)))
        else:
            modelo.add(Dense(unidades, activation="relu"))

        if DROPOUT_RATE > 0:
            modelo.add(Dropout(DROPOUT_RATE))

    modelo.add(Dense(1, activation="linear"))

    modelo.compile(optimizer=Adam(learning_rate=LEARNING_RATE),loss="mse")

    return modelo

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **6. Acompanhamento e monitoramento**
#
# <font color="blue"><b>Objetivos</b></font>:
# O callback personalizado permite acompanhar visualmente o processo de treinamento a cada época.
#
# São apresentados, para os conjuntos de treinamento e validação, os valores de Loss, MAE e R², além do tempo de execução e da melhor época identificada com base na Loss de validação.
#
# O painel é atualizado durante o treinamento de cada rede e de cada fold.
#
# **OBS.:** Esse acompanhamento é opcional e pode ser adaptado ou substituído de acordo com a preferência do usuário. A ausência desse painel não interfere no treinamento ou na avaliação das redes.

########################################################################
# CÉLULA DE CÓDIGO 21
# Célula 6
# Callback para monitoramento do treinamento, exibindo a cada época
# as métricas de desempenho, o tempo de execução e a melhor época.

class BarrasMetricasCallback(Callback):

    def __init__(self, X_train, y_train, X_val, y_val, nome="Rede"):
        super().__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.nome = nome
        self.painel = None

    def on_train_begin(self, logs=None):
        self.inicio = time.time()
        self.melhor_epoca = 0
        self.melhor_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        pred_train = self.model.predict(self.X_train, verbose=0).ravel()

        pred_val = self.model.predict(self.X_val, verbose=0).ravel()

        mae_train = mean_absolute_error(self.y_train, pred_train)
        mae_val = mean_absolute_error(self.y_val, pred_val)

        r2_train = r2_score(self.y_train, pred_train)
        r2_val = r2_score(self.y_val, pred_val)

        loss_train = logs.get("loss", 0)
        loss_val = logs.get("val_loss", 0)

        if loss_val < self.melhor_loss:
            self.melhor_loss = loss_val
            self.melhor_epoca = epoch + 1

        tempo = time.time() - self.inicio

        def barra(valor, escala, cor):
            largura = min(100, max(0, 100 * valor / escala))

            return f"""
            <div style="background:#e5e7eb; border-radius:6px; height:14px; width:100%;">
                <div style="width:{largura:.1f}%; background:{cor}; height:14px; border-radius:6px;"></div>
            </div>
            """

        html = f"""
        <div style="border:1px solid #d0d0d0; border-radius:8px; padding:14px; font-family:Arial;">

        <h3 style="margin-top:0;">{self.nome}</h3>

        <b>Época:</b> {epoch + 1} / {EPOCHS}
        {barra(epoch + 1, EPOCHS, "#16a34a")}

        <b>Loss de treinamento:</b> {loss_train:.5f}
        {barra(loss_train, 500, "#2563eb")}

        <b>Loss de validação:</b> {loss_val:.5f}
        {barra(loss_val, 500, "#7c3aed")}

        <b>MAE de treinamento:</b> {mae_train:.5f}
        {barra(mae_train, 25, "#0891b2")}

        <b>MAE de validação:</b> {mae_val:.5f}
        {barra(mae_val, 25, "#ea580c")}

        <b>R² de treinamento:</b> {r2_train:.5f}
        {barra(max(0, r2_train), 1, "#16a34a")}

        <b>R² de validação:</b> {r2_val:.5f}
        {barra(max(0, r2_val), 1, "#dc2626")}

        <p>
        <b>Tempo:</b> {tempo:.2f} s
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Melhor época:</b> {self.melhor_epoca}
        </p>

        </div>
        """

        # Cria o painel apenas na primeira época
        if self.painel is None:
            self.painel = display(HTML(html), display_id=True)
        else:
            # Atualiza o mesmo painel
            self.painel.update(HTML(html))

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **7. Treinamento e validação das arquiteturas**
#
# <font color="blue"><b>Objetivos</b></font>:
# Cada uma das oito arquiteturas é treinada utilizando os cinco folds do conjunto de desenvolvimento, totalizando 40 treinamentos.
#
# Em cada fold, a padronização das variáveis de entrada é realizada utilizando exclusivamente os dados do subconjunto de treinamento. O mesmo transformador é então aplicado ao subconjunto de validação.
#
# Após o treinamento, são calculadas as métricas R², MAE e MSE para cada fold. Ao final dos cinco folds, são calculadas a média e o desvio-padrão das métricas de cada arquitetura.
#
# O conjunto de teste final não é utilizado nesta etapa.

########################################################################
# CÉLULA DE CÓDIGO 23
# Célula 7
# Avaliação das 8 arquiteturas por validação cruzada (5 folds).
# Em cada fold, os dados são padronizados, a rede é treinada e avaliada,
# e as métricas são posteriormente agregadas por arquitetura.


# Resultados das 8 arquiteturas
resultados = []

for nome_rede, arquitetura in ARQUITETURAS.items():

    print("\n" + "="*60)
    print(f"{nome_rede} — Arquitetura: 29-{ '-'.join(map(str, arquitetura)) }-1")
    print("="*60)

    resultados_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev), start=1):

        print(f"\nTreinando {nome_rede} — Fold {fold}/{N_SPLITS}")

        # -------------------------------------------------
        # Separação do fold
        # -------------------------------------------------

        X_train_fold = X_dev.iloc[train_idx]
        X_val_fold = X_dev.iloc[val_idx]

        y_train_fold = y_dev.iloc[train_idx]
        y_val_fold = y_dev.iloc[val_idx]

        # -------------------------------------------------
        # Padronização dentro do fold
        # -------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # -------------------------------------------------
        # Criar modelo
        # -------------------------------------------------

        modelo = criar_modelo(input_dim=X_train_scaled.shape[1], arquitetura=arquitetura)


        # -------------------------------------------------
        # Callback gráfico
        # -------------------------------------------------

        callback = BarrasMetricasCallback(X_train_scaled, y_train_fold, X_val_scaled, y_val_fold, nome=f"{nome_rede} — Fold {fold}")


        # -------------------------------------------------
        # Early Stopping
        # -------------------------------------------------

        early_stopping = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)


        # -------------------------------------------------
        # Treinamento
        # -------------------------------------------------

        historico = modelo.fit(X_train_scaled, y_train_fold, validation_data=(X_val_scaled, y_val_fold), epochs=EPOCHS, batch_size=BATCH_SIZE,
                               callbacks=[callback, early_stopping], verbose=0)

        # -------------------------------------------------
        # Previsão no conjunto de validação
        # -------------------------------------------------

        y_pred = modelo.predict(X_val_scaled,verbose=0).ravel()

        # -------------------------------------------------
        # Métricas do fold
        # -------------------------------------------------

        r2 = r2_score(y_val_fold, y_pred)
        mae = mean_absolute_error(y_val_fold, y_pred)
        mse = mean_squared_error(y_val_fold, y_pred)

        epocas = len(historico.history["loss"])
        tempo = time.time() - callback.inicio

        resultados_folds.append({
            "R2": r2,
            "MAE": mae,
            "MSE": mse,
            "Epocas": epocas,
            "Tempo": tempo
        })

        print(f"Fold {fold}: "f"R² = {r2:.4f} | "f"MAE = {mae:.4f} | "f"MSE = {mse:.4f} | "f"Épocas = {epocas}")

    # =====================================================
    # MÉDIA DOS 5 FOLDS
    # =====================================================

    df_folds = pd.DataFrame(resultados_folds)

    resultados.append({
        "Rede": nome_rede, "Arquitetura": f"29-{ '-'.join(map(str, arquitetura)) }-1",

        "R² médio": df_folds["R2"].mean(), "R² DP": df_folds["R2"].std(),

        "MAE médio": df_folds["MAE"].mean(), "MAE DP": df_folds["MAE"].std(),

        "MSE médio": df_folds["MSE"].mean(), "MSE DP": df_folds["MSE"].std(),

        "Épocas médias": df_folds["Epocas"].mean(), "Tempo médio (s)": df_folds["Tempo"].mean()})

    # -----------------------------------------------------
    # Resultado da arquitetura
    # -----------------------------------------------------

    ultimo = resultados[-1]

    print("\n" + "-"*60)
    print(f"RESULTADO MÉDIO — {nome_rede}")
    print("-"*60)

    print(f"R²   = {ultimo['R² médio']:.4f} ± "f"{ultimo['R² DP']:.4f}")

    print(f"MAE  = {ultimo['MAE médio']:.4f} ± "f"{ultimo['MAE DP']:.4f}")

    print(f"MSE  = {ultimo['MSE médio']:.4f} ± "f"{ultimo['MSE DP']:.4f}")

    print(f"Épocas médias = {ultimo['Épocas médias']:.1f}")

    print(f"Tempo médio = {ultimo['Tempo médio (s)']:.2f} s")

########################################################################
# MARKDOWN / DOCUMENTAÇÃO
# **8. Impressão dos resultados**

########################################################################
# CÉLULA DE CÓDIGO 25
# Célula 8

df_resultados = pd.DataFrame(resultados)

df_resultados
