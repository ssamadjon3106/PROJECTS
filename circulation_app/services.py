from __future__ import annotations

from django.db import transaction
from django.utils import timezone

from library.models import Book, Loan


def loan_status(loan: Loan) -> str:
    if loan.returned_at:
        return 'returned'
    if loan.is_overdue:
        return 'overdue'
    return 'active'


def checkout_book_for_user(user, isbn: str) -> dict:
    """
    Performs atomic checkout:
    - validates book exists
    - ensures available_copies > 0
    - prevents multiple active loans for same user+book
    Returns a dict:
      - {'ok': True, 'book': Book}
      - {'ok': False, 'error': 'not_found'|'unavailable'|'already_checked_out', 'book': Book|None}
    """
    with transaction.atomic():
        try:
            book = Book.objects.select_for_update().get(isbn=isbn)
        except Book.DoesNotExist:
            return {'ok': False, 'error': 'not_found', 'book': None}

        if book.available_copies < 1:
            return {'ok': False, 'error': 'unavailable', 'book': book}

        if Loan.objects.filter(user=user, book=book, returned_at__isnull=True).exists():
            return {'ok': False, 'error': 'already_checked_out', 'book': book}

        book.available_copies -= 1
        book.save(update_fields=['available_copies'])
        Loan.objects.create(user=user, book=book)

        return {'ok': True, 'book': book}


def checkin_loan_for_user(user, loan_id: int) -> dict:
    """
    Performs atomic check-in:
    - ensures loan belongs to user and is currently active
    - marks loan returned_at
    - increments book available_copies (up to total_copies)
    Returns a dict:
      - {'ok': True, 'book': Book}
      - {'ok': False, 'error': 'not_found'}
    """
    with transaction.atomic():
        loan = (
            Loan.objects.select_related('book')
            .select_for_update()
            .filter(id=loan_id, user=user, returned_at__isnull=True)
            .first()
        )
        if loan is None:
            return {'ok': False, 'error': 'not_found'}

        loan.returned_at = timezone.now()
        loan.save(update_fields=['returned_at'])

        book = Book.objects.select_for_update().get(id=loan.book_id)
        if book.available_copies < book.total_copies:
            book.available_copies += 1
            book.save(update_fields=['available_copies'])

        return {'ok': True, 'book': book}


def get_my_loans_summary(user) -> dict:
    loans = Loan.objects.select_related('book').filter(user=user)
    active_count = loans.filter(returned_at__isnull=True).count()
    returned_count = loans.filter(returned_at__isnull=False).count()
    overdue_count = sum(loan.is_overdue for loan in loans)

    return {
        'loans': loans,
        'active_count': active_count,
        'returned_count': returned_count,
        'overdue_count': overdue_count,
    }
