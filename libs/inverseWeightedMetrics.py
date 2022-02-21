from collections import defaultdict

def precision_recall_f1_support(y_test, y_pred):
    # Conta o numero de true positives e false negatives, por categoria
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for idx, value in enumerate(y_test):
        if value == y_pred[idx]:
            true_positives[value] += 1
        else:
            false_positives[y_pred[idx]] += 1
            false_negatives[value] += 1
    
    # Calcula as métricas, por classe
    class_metrics = {}

    for key, val in true_positives.items():
        support = y_test.count(key)
        precision = val / (val + false_positives[key])
        recall = val / support
        f1_score = (2*precision*recall) / (precision + recall)

        class_metrics[key] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support   
        }

    return class_metrics

def inverse_weighted_average(valueWeights):
    '''
        Recebe uma lista de tuplas, contendo os valores e pesos
        Retorna a média ponderada invertida dos valores
    '''

    #TODO: Checar se é lista de tuplas

    total_value = 0
    total_weight = 0
    
    for value,weight in valueWeights:
        inverse_value = value / weight
        total_value += inverse_value
        total_weight += 1 / weight

    return total_value / total_weight

def calculate_inverse_weighted_metrics(y_test, y_pred):

    y_test_list = [*y_test]
    y_pred_list = [*y_pred]

    class_metrics = precision_recall_f1_support(y_test_list, y_pred_list)

    precision_values = [(c['precision'], c['support'])
                        for c in class_metrics.values()]

    recall_values = [(c['recall'], c['support'])
                        for c in class_metrics.values()]

    f1_score_values = [(c['f1_score'], c['support'])
                        for c in class_metrics.values()]

    precision = inverse_weighted_average(precision_values)
    recall = inverse_weighted_average(recall_values)
    f1_score = inverse_weighted_average(f1_score_values)

    return precision, recall, f1_score
