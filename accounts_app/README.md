# Accounts App

Owner: Team Member 1

Purpose: student registration, login experience, profile data, and session handling.

Architecture:

- `models.py` stores future profile/student metadata.
- `forms.py` validates account and profile input.
- `views.py` renders account pages.
- `services.py` contains account workflow logic.
- `selectors.py` contains account query helpers.
- `urls.py` owns `/team/accounts/`.
- `templates/accounts_app/` owns account templates.
- `static/accounts_app/` owns account-specific assets.
