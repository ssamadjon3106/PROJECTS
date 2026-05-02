from django.db.models import Q, Sum

from circulation_app.selectors import active_loans
from library.models import Book


def search_books(*, query: str, selected_category: str, available_only: bool):
    books = Book.objects.all()

    if query:
        books = books.filter(
            Q(title__icontains=query)
            | Q(author__icontains=query)
            | Q(isbn__icontains=query)
            | Q(category__icontains=query)
        )

    if selected_category:
        books = books.filter(category=selected_category)

    if available_only:
        books = books.filter(available_copies__gt=0)

    return books


def categories_list():
    return (
        Book.objects.exclude(category='')
        .values_list('category', flat=True)
        .distinct()
        .order_by('category')
    )


def books_copy_stats():
    all_books = Book.objects.all()
    total_copies = all_books.aggregate(total=Sum('total_copies'))['total'] or 0
    available_copies = all_books.aggregate(total=Sum('available_copies'))['total'] or 0

    return {
        'titles': all_books.count(),
        'total_copies': total_copies,
        'available_copies': available_copies,
    }


def overdue_count_for_user(user) -> int:
    loans = active_loans().filter(user=user)
    return sum(loan.is_overdue for loan in loans)


def get_active_loans_for_user(user):
    return active_loans().filter(user=user)


def user_has_active_loan(*, user, book) -> bool:
    if not user.is_authenticated:
        return False
    return active_loans().filter(user=user, book=book).exists()
