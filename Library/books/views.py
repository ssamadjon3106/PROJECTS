from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
from django.conf import settings
from .models import Book, BorrowRecord


def home_view(request):
    books = Book.objects.all()
    return render(request, 'books/home.html', {'books': books})


def available_books_view(request):
    books = Book.objects.filter(available_copies__gt=0)
    return render(request, 'books/home.html', {'books': books})


def detail_view(request, pk):
    book = get_object_or_404(Book, pk=pk)
    return render(request, 'books/detail.html', {'book': book})


@login_required
def borrow_view(request, pk):
    book = get_object_or_404(Book, pk=pk)
    if not book.is_available():
        messages.error(request, 'No copies available.')
        return redirect('books:detail', pk=pk)
    already = BorrowRecord.objects.filter(user=request.user, book=book, status=BorrowRecord.ACTIVE).exists()
    if already:
        messages.warning(request, 'You already borrowed this book.')
        return redirect('books:detail', pk=pk)
    due_date = timezone.now().date() + timedelta(days=settings.MAX_BORROW_DAYS)
    BorrowRecord.objects.create(user=request.user, book=book, due_date=due_date)
    book.available_copies -= 1
    book.save()
    request.user.points += 10
    request.user.save()
    messages.success(request, f'Borrowed! Due: {due_date}')
    return redirect('books:my_books')


@login_required
def return_view(request, pk):
    record = get_object_or_404(BorrowRecord, pk=pk, user=request.user)
    if record.status == BorrowRecord.RETURNED:
        messages.warning(request, 'Already returned.')
        return redirect('books:my_books')
    record.status = BorrowRecord.RETURNED
    record.returned_at = timezone.now()
    record.save()
    record.book.available_copies += 1
    record.book.save()
    if record.days_overdue() == 0:
        request.user.points += 20
        request.user.save()
    messages.success(request, 'Book returned!')
    return redirect('books:my_books')


@login_required
def my_books_view(request):
    records = BorrowRecord.objects.filter(user=request.user).order_by('-borrowed_at')
    return render(request, 'books/my_books.html', {'records': records})
