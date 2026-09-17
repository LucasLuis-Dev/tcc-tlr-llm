"""
Módulo para avaliação das predições de TLR (Precision, Recall, F-score, MAP, etc.).
"""
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de classificação padrão para validação dos links de rastreabilidade.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
