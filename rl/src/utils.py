import numpy as np


def two_combinations(vector):
    """
    Convert an n-value vector to an n^2 2-dimensional vector representing all 2-combinations.

    Examples:

    >>> two_combinations([1, 2, 3])
    >>> [[1 1]
        [1 2]
        [1 3]
        [2 1]
        [2 2]
        [2 3]
        [3 1]
        [3 2]
        [3 3]]

    :param vector: An n-value vector (list or numpy array).
    :return: An n^2 2-dimensional vector (2D numpy array).
    """
    n = len(vector)
    combination_vectors = []

    for i in range(n):
        for j in range(n):
            combination_vectors.append([vector[i], vector[j]])

    return np.array(combination_vectors)
