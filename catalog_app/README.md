# Catalog App

Owner: Team Member 2

Purpose: book inventory, search, categories, Excel imports, and catalog presentation.

Architecture:

- `models.py` stores future catalog-specific models.
- `forms.py` validates search and import input.
- `views.py` renders catalog pages.
- `services.py` contains import/indexing workflows.
- `selectors.py` contains book query helpers.
- `urls.py` owns `/team/catalog/`.
- `templates/catalog_app/` owns catalog templates.
- `static/catalog_app/` owns catalog-specific assets.
