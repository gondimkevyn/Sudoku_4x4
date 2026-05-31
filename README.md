# 🧩 Solução de Sudoku 4x4 com Rede Neural Artificial (MLP)

Este repositório contém o desenvolvimento de uma solução de Inteligência Artificial baseada em uma **Rede Neural Artificial (RNA) Multicamadas (Multi-Layer Perceptron - MLP)** desenvolvida em Python com Keras/TensorFlow para resolver quebra-cabeças de Sudoku $4 \times 4$.

O projeto aborda a modelagem do tabuleiro como um problema de classificação multiclasse multilabel, avalia o comportamento da rede frente a cenários de maior dificuldade e traz uma análise crítica sobre os limites do aprendizado profundo em problemas de raciocínio lógico.

---

## 🚀 Estrutura do Projeto

O código foi componentizado em três partes principais para garantir organização e fácil manutenção:

1. **Módulo do Gerador de Dados (`SudokuDataGenerator`):** Cria de forma dinâmica um conjunto de dados baseado em **4 modelos estruturais de tabuleiros totalmente diferentes** e perfeitamente válidos. Para elevar o nível de dificuldade, o gerador aplica uma máscara que oculta estritamente **entre 6 e 10 elementos** de cada grade de teste.
2. **Módulo da Rede Neural Artificial (`SudokuMLPModel`):** Define uma arquitetura perceptron multicamadas profunda. Utiliza camadas densas (`Dense`), normalização por lote (`BatchNormalization`) para estabilidade, e `Dropout` para evitar memorização excessiva (*overfitting*). A camada de saída utiliza a função `Softmax` aplicada individualmente em cada uma das 16 células para prever a probabilidade dos números de $S = \{1, 2, 3, 4\}$.
3. **Pipeline de Orquestração e Validação:** Embaralha e divide os dados (80% treino / 20% teste), realiza o treinamento e seleciona **4 tabuleiros iniciais aleatórios e estritamente únicos** para demonstrar a capacidade de convergência e resolução da rede em tempo de execução.

---

## 🧠 Análise Teórica: Limitações e o Desafio $N \times N$

### 1. O Problema da Generalização ($4 \times 4$ para $N \times N$)
Uma rede MLP treinada exclusivamente para instâncias $4 \times 4$ **não consegue generalizar** para tabuleiros maiores (como o clássico $9 \times 9$) devido a travas arquiteturais:
* **Rigidez Dimensional:** O número de neurônios de entrada e saída é fixo. Mudar para $9 \times 9$ exige alterar a entrada de 16 para 81 neurônios e a saída de 64 para 729 neurônios, quebrando toda a estrutura de pesos já aprendida.
* **Explosão Combinatória:** Enquanto o espaço amostral do Sudoku $4 \times 4$ possui apenas 288 soluções únicas, o modelo $9 \times 9$ escala para $\approx 6.67 \times 10^{21}$ combinações possíveis, tornando o mapeamento por aproximação estatística de força bruta inviável.

### 2. Amostragem Estatística vs. Raciocínio Lógico Clássico
**Qual o problema em gerar amostras e testá-las, se isso é tratado como um problema de raciocínio?**

O erro conceitual ao tentar resolver o Sudoku puro via Redes Neurais padrão (MLP/CNN) reside no fato de que **RNAs são aproximadores estatísticos**, e não motores de inferência lógica:
* **Ausência de Regras Abstratas:** A rede não aprende a regra lógica (ex: *"se o número 3 está na coluna, não pode se repetir"*). Ela decora correlações estatísticas de vizinhança espacial presentes no dataset de treino. Se o puzzle exigir múltiplos passos de dedução encadeada (*look-ahead*), a rede falha por tentar preencher tudo simultaneamente com base no peso de maior probabilidade.
* **Falta de Validação de Restrições Nativas:** A amostragem pura não garante consistência matemática. Sem uma camada dedicada de verificação, a rede pode propor números duplicados se a confiança individual daqueles neurônios for alta, pois ela carece de um mecanismo nativo de *backtracking* (tentativa e erro com retorno).

> 💡 **Como a IA resolve o raciocínio real?** Para tratar o Sudoku estruturalmente como raciocínio, o mercado utiliza **Redes Neurais de Grafos (GNNs)** (onde células são nós e restrições são arestas trocando mensagens de consistência) ou abordagens determinísticas como **Programação por Restrições (Constraint Programming)** e **SAT Solvers**.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy**

---

## 🎯 Resultados Esperados

Ao executar o pipeline, o modelo atinge rapidamente alta acurácia e exibe no terminal o fluxo completo para os 4 casos isolados:

```text
================ CASO TESTE DE SUDOKU #1 ================
Quantidade de lacunas para preencher: 8

Tabuleiro Inicial:
[[3 4 0 0]
 [0 2 0 0]
 [0 3 2 0]
 [2 0 4 3]]

Solução Proposta pela Rede Neural:
[[3 4 1 2]
 [1 2 3 4]
 [4 3 2 1]
 [2 1 4 3]]

================ CASO TESTE DE SUDOKU #2 ================
Quantidade de lacunas para preencher: 10

Tabuleiro Inicial:
[[4 3 0 1]
 [0 0 0 3]
 [0 0 1 0]
 [0 2 0 0]]

Solução Proposta pela Rede Neural:
[[4 3 2 1]
 [2 1 4 3]
 [3 4 1 2]
 [1 2 3 4]]

================ CASO TESTE DE SUDOKU #3 ================
Quantidade de lacunas para preencher: 7

Tabuleiro Inicial:
[[0 0 2 0]
 [0 1 4 0]
 [0 4 0 2]
 [1 2 3 4]]

Solução Proposta pela Rede Neural:
[[4 3 2 1]
 [2 1 4 3]
 [3 4 1 2]
 [1 2 3 4]]

================ CASO TESTE DE SUDOKU #4 ================
Quantidade de lacunas para preencher: 6

Tabuleiro Inicial:
[[0 4 1 2]
 [1 0 0 4]
 [4 0 0 0]
 [2 1 4 3]]

Solução Proposta pela Rede Neural:
[[3 4 1 2]
 [1 2 3 4]
 [4 3 2 1]
 [2 1 4 3]]

