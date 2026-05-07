from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.utils import timezone
from books.models import Book, BorrowRecord
from seats.models import Seat, Reservation
from accounts.models import LibraryUser


def librarian_required(view_func):
    @login_required
    def wrapper(request, *args, **kwargs):
        if not request.user.is_librarian():
            raise PermissionDenied
        return view_func(request, *args, **kwargs)
    return wrapper


@librarian_required
def home_view(request):
    today = timezone.now().date()

    total_books = Book.objects.count()
    total_students = LibraryUser.objects.filter(role=LibraryUser.STUDENT).count()
    active_borrows = BorrowRecord.objects.filter(status=BorrowRecord.ACTIVE).count()
    active_reservations = Reservation.objects.filter(status=Reservation.ACTIVE).count()

    overdue_records = BorrowRecord.objects.filter(
        status=BorrowRecord.ACTIVE,
        due_date__lt=today
    ).select_related('user', 'book').order_by('due_date')

    context = {
        'total_books': total_books,
        'total_students': total_students,
        'active_borrows': active_borrows,
        'active_reservations': active_reservations,
        'overdue_records': overdue_records,
        'overdue_count': overdue_records.count(),
    }
    return render(request, 'dashboard/home.html', context)


@librarian_required
def overdue_view(request):
    today = timezone.now().date()
    records = BorrowRecord.objects.filter(
        status=BorrowRecord.ACTIVE,
        due_date__lt=today
    ).select_related('user', 'book').order_by('due_date')
    return render(request, 'dashboard/overdue.html', {'records': records})


@librarian_required
def seats_view(request):
    seats = Seat.objects.all().prefetch_related('reservations')
    return render(request, 'dashboard/seats.html', {'seats': seats})