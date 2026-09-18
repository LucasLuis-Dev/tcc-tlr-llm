"""
Módulo para avaliação das predições de TLR (Precision, Recall, F1-score, Matriz de Confusão).
Utiliza a biblioteca scikit-learn para cálculo formal de métricas de Recuperação de Informação.
"""
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Calcula e exibe no terminal as métricas clássicas de Recuperação de Informação (TLR):
    Precisão, Revocação, F1-Score e Matriz de Confusão.

    :param y_true: Lista com os valores reais do gabarito (1 para elo existente, 0 para sem elo).
    :param y_pred: Lista com as predições numéricas da IA (1 para SIM, 0 para NÃO).
    :return: Dicionário com as métricas calculadas.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    print("\n" + "=" * 60)
    print("        RESUMO DE MÉTRICAS DE AVALIAÇÃO (TLR)")
    print("=" * 60)
    print(f"Total de pares avaliados : {total}")
    print(f"Precisão (Precision)     : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Revocação (Recall)        : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score                 : {f1:.4f} ({f1 * 100:.2f}%)")
    print("-" * 60)
    print("Matriz de Confusão:")
    print(f"                 Predito: NÃO (0)     Predito: SIM (1)")
    print(f"  Real: NÃO (0)       {tn:<20} {fp:<20} (Total Neg: {tn + fp})")
    print(f"  Real: SIM (1)       {fn:<20} {tp:<20} (Total Pos: {fn + tp})")
    print("-" * 60)
    print(f"  Verdadeiros Positivos (TP) : {tp}")
    print(f"  Falsos Positivos (FP)      : {fp}")
    print(f"  Verdadeiros Negativos (TN) : {tn}")
    print(f"  Falsos Negativos (FN)      : {fn}")
    print("=" * 60 + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    }


if __name__ == "__main__":
    # Teste rápido com dados sintéticos
    y_test_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_test_pred = [1, 0, 1, 0, 0, 1, 1, 0]
    calculate_metrics(y_test_true, y_test_pred)
