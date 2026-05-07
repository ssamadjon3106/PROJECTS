from django.db import models
from django.conf import settings
from django.utils import timezone
from datetime import timedelta


class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=200)
    isbn = models.CharField(max_length=20, unique=True)
    genre = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    total_copies = models.PositiveIntegerField(default=1)
    available_copies = models.PositiveIntegerField(default=1)

    def is_available(self):
        return self.available_copies > 0

    def __str__(self):
        return f'{self.title} by {self.author}'


class BorrowRecord(models.Model):
    ACTIVE = 'active'
    RETURNED = 'returned'
    STATUS_CHOICES = [
        (ACTIVE, 'Active'),
        (RETURNED, 'Returned'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='borrow_records')
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='borrow_records')
    borrowed_at = models.DateTimeField(auto_now_add=True)
    due_date = models.DateField()
    returned_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=ACTIVE)

    def days_overdue(self):
        if self.status == self.RETURNED:
            return 0
        return max(0, (timezone.now().date() - self.due_date).days)

    def fine(self):
        return self.days_overdue() * settings.FINE_PER_DAY

    def __str__(self):
        return f'{self.user.username} — {self.book.title}'