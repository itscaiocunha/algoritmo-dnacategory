"""Classificação binária (gene tem ou não a função alvo) com SVM linear em k-mers."""
import warnings
from dataclasses import dataclass

from imblearn.under_sampling import RandomUnderSampler
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .features import sequencias_para_kmers

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


@dataclass
class ResultadoTreino:
    modelo: SVC
    vectorizer: CountVectorizer
    y_test: object
    y_pred: object
    y_scores: object  # decision_function, usado na curva ROC
    acuracia: float


def balancear_undersampling(df_flat, random_state=42):
    """Reduz a classe majoritária até ficar 1:1 com a minoritária."""
    rus = RandomUnderSampler(random_state=random_state)
    indices = df_flat.index.values.reshape(-1, 1)
    indices_res, _ = rus.fit_resample(indices, df_flat['label'])
    return df_flat.loc[indices_res.flatten()].copy()


def treinar_svm(df_flat, kmer_size, test_size=0.2, C=1.0, random_state=42):
    """Treina e avalia um SVM linear no dataset achatado (colunas 'Sequencia', 'label')."""
    print(f"Distribuição ANTES do balanceamento:\n{df_flat['label'].value_counts()}\n")
    df_balanced = balancear_undersampling(df_flat, random_state)
    print(f"Distribuição DEPOIS do balanceamento:\n{df_balanced['label'].value_counts()}\n")

    print(f"Vetorizando k-mers (k={kmer_size})...")
    # Vocabulário aprendido em todos os genes, não só nos que sobraram do balanceamento
    vectorizer = CountVectorizer(analyzer='word')
    vectorizer.fit(sequencias_para_kmers(df_flat['Sequencia'], kmer_size))
    X = vectorizer.transform(sequencias_para_kmers(df_balanced['Sequencia'], kmer_size))
    y = df_balanced['label']
    print(f"Matriz de features X: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Treinando SVM (kernel='linear') em {len(y_train)} amostras...")
    modelo = SVC(kernel='linear', C=C, random_state=random_state)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia: {acuracia * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    return ResultadoTreino(
        modelo=modelo,
        vectorizer=vectorizer,
        y_test=y_test,
        y_pred=y_pred,
        y_scores=modelo.decision_function(X_test),
        acuracia=acuracia,
    )
