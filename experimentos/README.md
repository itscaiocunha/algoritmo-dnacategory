# Histórico de experimentos

Cada pasta guarda o código exatamente como foi usado naquela iteração. Só os
caminhos de arquivo foram ajustados para a estrutura atual. O pipeline
consolidado está em [`src/dnacategory`](../src/dnacategory).

Para rodar um experimento, entre na pasta dele (os caminhos são relativos):

```bash
cd experimentos/v04_imputacao_knn_ponderado
python main.py
```

## Parte 1: imputação de bases `N`

Validação por simulação: sequências limpas são divididas em referência (80 %)
e teste (20 %). Em cada sequência de teste, 5 % das posições são trocadas por
`N`, e mede-se a acurácia do imputador em recuperar a base original.

A numeração das iterações segue o relatório de pesquisa.

| Versão | Iteração | Abordagem | O que mudou | Acurácia |
|--------|----------|-----------|-------------|----------|
| [v01](v01_imputacao_knn_global) | 1 (baseline) | KNN global | Busca os k = 5 vizinhos (Hamming) em toda a referência; voto majoritário simples. Amostra de 1.500 genes. | 26,60 % |
| [v02](v02_imputacao_kmeans_knn) | 2 e 3 | K-Means + KNN | Agrupa os genes por k-mers e busca vizinhos só dentro do cluster. A iteração 2 usou k = 4 e 50 clusters (56,03 %). O arquivo está com a configuração da iteração 3: k = 6, 200 clusters. | 58,39 % |
| [v03](v03_imputacao_tfidf) | 4.1 | TF-IDF + K-Means + KNN | Troca a contagem de k-mers por TF-IDF na clusterização. | 57,92 % |
| [v04](v04_imputacao_knn_ponderado) | 4.2 (campeão) | K-Means + KNN ponderado | Volta ao CountVectorizer; o voto passa a ter peso `1 / (distância + 1)`. **Abordagem adotada no pipeline.** | **62,73 %** |

## Parte 2: classificação de função com SVM

Classificação binária: `protein binding` (1) contra outras funções (0).

| Versão | Abordagem | O que mudou |
|--------|-----------|-------------|
| [v05](v05_svm_class_weight) | Imputador em produção + SVM com `class_weight='balanced'` | Imputador separado em treino (`treinar_imputador.py`) e aplicação em chunks (`aplicar_imputador.py`). Primeiro protótipo do SVM, em um subconjunto do dataset mestre. |
| [v06](v06_svm_undersampling) | SVM + undersampling manual | Balanceia 1:1 descartando amostras da classe majoritária. |
| [v07](v07_svm_smote) | SVM + SMOTE | Oversampling sintético aplicado **apenas no treino**, depois da divisão; avaliação no teste com a distribuição real. |
| [v08](v08_svm_dataset_completo) | Dataset completo + checkpoint | Imputação do dataset mestre inteiro (~14,4M linhas, 2.884 lotes), com checkpoint para retomar execuções interrompidas. SVM com undersampling: 5.014 genes (3.723 contra 1.291) balanceados para 1.291 contra 1.291. |
| [v09](v09_svm_graficos) | Dataset achatado + gráficos | Treina sobre o dataset já achatado (1 linha por gene), usa `RandomUnderSampler` e gera matriz de confusão e curva ROC. **Base do pipeline atual.** |

### Anotações dos protótipos (subconjuntos do dataset mestre)

| Linhas do dataset mestre | Tempo | Acurácia |
|--------------------------|-------|----------|
| 5 mil   | ~1 min | 75,00 % |
| 25 mil  | ~1 min | 69,81 % |
| 50 mil  | ~1 min | 76,53 % |
| 100 mil | ~1 min | 72,22 % |

> Essas acurácias foram medidas em subconjuntos pequenos e, em parte, sem
> balanceamento do teste. Elas não são comparáveis diretamente com os
> resultados no dataset completo: 59,72 % e AUC 0,595 reportados no
> relatório; 53,8 % e AUC 0,554 na execução da v09 que gerou os gráficos em
> `reports/figures/`.

## Parte 3: Deep Learning ([v10](v10_deep_learning))

Notebooks executados no Google Colab (GPU Tesla T4, TensorFlow/Keras), com
as saídas originais preservadas. Para rodar de novo, abra o notebook no
Colab e ajuste o caminho do Google Drive na célula de configuração.

Todos usam o mesmo protocolo do SVM: undersampling 1:1 com
`RandomUnderSampler(random_state=42)` e divisão 80/20 estratificada com
`random_state=42` (517 genes de teste). Por isso os resultados são
comparáveis. As sequências são tokenizadas por caractere e **truncadas ou
completadas com padding até 2.000 bases**. Em seguida passam por uma camada
`Embedding` de 100 dimensões.

| Notebook | Modelo | Dados | Acurácia | Recall (0 / 1) |
|----------|--------|-------|----------|----------------|
| [CNN-versao01](v10_deep_learning/CNN-versao01.ipynb) | Protótipo: Conv1D(64) + Flatten, 10 épocas | Amostra de 500 mil linhas do dataset mestre (3.253 genes, 682 + 682) | 54,95 % | 0,49 / 0,61 |
| [CNN-versao02](v10_deep_learning/CNN-versao02.ipynb) | 2 blocos Conv1D + GlobalMaxPooling + EarlyStopping | Tentativa com o dataset achatado | 57,06 %* | 0,83 / 0,31 |
| [CNN-versao03](v10_deep_learning/CNN-versao03.ipynb) | Mesma arquitetura da v02 | Dataset achatado (5.014 genes, 1.291 + 1.291) | **59,19 %** | 0,58 / 0,60 |
| [CNN-versao03](v10_deep_learning/CNN-versao03.ipynb) (última célula) | MLP: Dense(128) → Dense(64), k-mers k = 6 + StandardScaler | Dataset achatado | 58,99 %** | 0,53 / 0,66** |
| [CNN-LSTM](v10_deep_learning/CNN-LSTM.ipynb) | Conv1D(64) + LSTM bidirecional(100) | Dataset achatado | **59,57 %** | 0,53 / 0,66 |

\* Na CNN-versao02, a célula de carga falhou (`ValueError`: a coluna
`GO term name` não existe no dataset achatado). As saídas das células
seguintes vêm de uma execução anterior da mesma sessão. A CNN-versao03
corrige a carga.

\** A célula do MLP não tem saída salva no notebook. O valor é o do relatório
de pesquisa.

**Observações:**
- O protótipo CNN-versao01 mostra overfitting forte: 94 % de acurácia no
  treino contra 55 % na validação. As versões seguintes adicionam Dropout,
  GlobalMaxPooling e EarlyStopping.
- O `EarlyStopping` monitora a perda no próprio conjunto de teste
  (`validation_data=(X_test, y_test)`). O teste acaba participando da
  escolha da época, o que deixa as acurácias das redes levemente otimistas.
