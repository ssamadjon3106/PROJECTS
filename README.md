# LibraryHub — SE201 Web Programming Capstone

A library management system for students and staff of New Uzbekistan University.

## Team — Piyanshikler

| Member | App | Responsibility |
|--------|-----|----------------|
| Nurlan | `accounts_app` | User registration, login, logout, profile data, and session policies |
| Samadjon | `library` + `circulation_app` | Book catalog + borrow/return (loans, due/overdue logic) |
| Aynazar | `seats_app` | Interactive floor map + seat reservations |
| Abror | `catalog_app` | Catalog/search presentation (title/category browsing, demo architecture) |
| Anvar | `reports_app` | Reports/analytics (demo architecture + placeholders) |

## What students can do

- Browse the catalog and open book details
- Check out books by ISBN and return active loans
- Reserve seats on the entrance-zone seat map

## What staff/admin can do

- Manage books, loans, seats, and seat reservations in the Django admin panel (`/admin/`)
- Staff can view reservation owner info (non-staff users don’t see who reserved seats)

---

## Project layout (actual repo structure)

At the repository root you have:

- `manage.py` — Django entrypoint
- `config/` — Django settings + root URL routing
- `library/` — shared models, shared views, and the main site routes (`dashboard`, `checkout`, `my loans`, `seats`)
- `accounts_app/` — auth UX (login/register/logout pages) and app architecture page
- `catalog_app/` — catalog demo architecture page
- `circulation_app/` — circulation demo architecture page (and checkout/check-in endpoints are wired through `library/urls.py`)
- `seats_app/` — seat map reservation UI and reservation logic (entrance-zone rules)
- `reports_app/` — reports demo architecture page
- `templates/` — project-wide templates (base layout + pages)

Run-time navigation is wired via `config/urls.py` and `library/urls.py`.

---

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python manage.py migrate
.venv/bin/python manage.py seed_library
.venv/bin/python manage.py import_books_xlsx "/Users/samadjon/Documents/Sophomore/Web_programming/Книга1.xlsx"
.venv/bin/python manage.py createsuperuser
.venv/bin/python manage.py runserver
```

Open: `http://127.0.0.1:8000/`

### Staff/Admin setup
To grant admin access in Django:
- Go to `/admin/`
- Find the user in **Users**
- Enable **staff status** (and/or make them superuser)

---

## Demo flow

1. Register a student account
2. Browse the catalog and open a book detail page
3. Checkout a book by ISBN, then return it via **My Loans**
4. Go to **Seats** and reserve an entrance-zone seat for a selected time slot
5. Use `/admin/` to manage books/loans/seats/reservations

---

## Team “architecture” pages

These are demo pages owned by each app:

- `/team/accounts/`  → `accounts_app`
- `/team/catalog/`   → `catalog_app`
- `/team/circulation/` → `circulation_app`
- `/team/seats/`      → `seats_app`
- `/team/reports/`    → `reports_app`
