from library.models import Loan


def active_loans():
    return Loan.objects.select_related('user', 'book').filter(returned_at__isnull=True)
