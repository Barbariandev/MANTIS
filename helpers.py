
def normalize_weights(weights_dict: dict[int, float]) -> dict[int, float]:
    """
    Normalize weights dictionary so that the sum equals 1.0.

    Args:
        weights_dict: Dictionary mapping UID to weight value
    Returns:
        Normalized weights dictionary with sum = 1.0
    """

    total = sum(weights_dict.values())
    if total <= 0:
        return {}
    return {uid: w / total for uid, w in weights_dict.items()}