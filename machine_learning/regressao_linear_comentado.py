
# Importa a biblioteca NumPy para trabalhar com vetores e arrays numéricos
import numpy as np

# Importa o módulo pyplot da biblioteca matplotlib para criar gráficos
import matplotlib.pyplot as plt

# Define os dados de entrada: tamanhos das casas em milhares de pés quadrados (1000 sqft)
# x_train representa os tamanhos das casas no conjunto de treino
x_train = np.array([1.0, 2.0])

# Define os preços reais das casas em milhares de dólares
# y_train representa os preços das casas no conjunto de treino
y_train = np.array([300.0, 500.0])

# Define os parâmetros do modelo: w é o peso (inclinação da reta) e b é o viés (intercepto)
# Esses valores foram escolhidos para que a linha passe exatamente pelos pontos dados
w = 200
b = 100

# Define uma função que calcula as previsões do modelo para cada entrada x
def compute_model_output(x, w, b):
    # x é um vetor com m exemplos; essa linha descobre quantos exemplos temos
    m = x.shape[0]

    # Cria um vetor vazio com m posições para guardar as previsões
    f_wb = np.zeros(m)

    # Para cada exemplo, calcula ŷ = w * x + b e armazena no vetor de previsões
    for i in range(m):
        f_wb[i] = w * x[i] + b

    # Retorna todas as previsões
    return f_wb

# Usa a função definida para calcular os valores previstos para os dados de treino
y_pred = compute_model_output(x_train, w, b)

# Cria um gráfico de dispersão (scatter) com os dados reais em vermelho com marca 'x'
plt.scatter(x_train, y_train, marker='x', c='r', label='Dados reais')

# Desenha a linha de previsão do modelo em azul
plt.plot(x_train, y_pred, c='b', label='Previsão do modelo')

# Adiciona um título ao gráfico
plt.title('Housing Prices - Regressão Linear')

# Define o rótulo do eixo x
plt.xlabel('Tamanho da casa (1000 sqft)')

# Define o rótulo do eixo y
plt.ylabel('Preço (em mil dólares)')

# Mostra a legenda para identificar os dados e a linha do modelo
plt.legend()

# Ativa a grade no fundo do gráfico
plt.grid(True)

# Ajusta o layout para evitar cortes nos rótulos
plt.tight_layout()

# Exibe o gráfico
plt.show()

# Agora vamos prever o preço de uma casa nova com 1200 sqft (ou seja, x = 1.2)
x_i = 1.2

# Aplica a fórmula da regressão linear: ŷ = w * x + b
cost = w * x_i + b

# Imprime o resultado da previsão formatado como inteiro (sem casas decimais)
print(f"Preço previsto para 1200 sqft: ${cost:.0f} mil dólares")
