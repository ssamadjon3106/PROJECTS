def normalize_isbn(isbn):
    return ''.join(str(isbn or '').split()).upper()
