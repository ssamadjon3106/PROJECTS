def percentage(part, whole):
    if not whole:
        return 0
    return round((part / whole) * 100, 1)
