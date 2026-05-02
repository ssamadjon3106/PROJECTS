from library.models import Book, Loan, SeatReservation


def dashboard_totals():
    return {
        'books': Book.objects.count(),
        'active_loans': Loan.objects.filter(returned_at__isnull=True).count(),
        'seat_reservations': SeatReservation.objects.count(),
    }
