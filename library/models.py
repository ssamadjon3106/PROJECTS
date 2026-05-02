from datetime import datetime, timedelta

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.utils import timezone


def default_due_date():
    return timezone.now() + timedelta(days=14)


class Book(models.Model):
    title = models.CharField(max_length=180)
    author = models.CharField(max_length=120)
    isbn = models.CharField('ISBN', max_length=20, unique=True)
    category = models.CharField(max_length=80, blank=True)
    shelf_location = models.CharField(max_length=40, blank=True)
    total_copies = models.PositiveIntegerField(default=1)
    available_copies = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['title', 'author']

    def __str__(self):
        return f'{self.title} by {self.author}'

    def clean(self):
        if self.available_copies > self.total_copies:
            raise ValidationError('Available copies cannot be greater than total copies.')

    @property
    def is_available(self):
        return self.available_copies > 0


class Loan(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='loans')
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='loans')
    checked_out_at = models.DateTimeField(auto_now_add=True)
    due_at = models.DateTimeField(default=default_due_date)
    returned_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-checked_out_at']
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'book'],
                condition=Q(returned_at__isnull=True),
                name='unique_active_book_loan_per_user',
            )
        ]

    def __str__(self):
        return f'{self.user} - {self.book}'

    @property
    def is_active(self):
        return self.returned_at is None

    @property
    def is_overdue(self):
        return self.is_active and self.due_at < timezone.now()

    @property
    def is_due_soon(self):
        if not self.is_active or self.is_overdue:
            return False
        return self.due_at <= timezone.now() + timedelta(days=3)


class Seat(models.Model):
    label = models.CharField(max_length=10, unique=True)
    zone = models.CharField(max_length=60)
    row = models.PositiveSmallIntegerField(default=1)
    column = models.PositiveSmallIntegerField(default=1)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['zone', 'label']

    def __str__(self):
        return f'{self.label} - {self.zone}'


class SeatReservation(models.Model):
    STATUS_ACTIVE = 'active'
    STATUS_CANCELLED = 'cancelled'

    STATUS_CHOICES = [
        (STATUS_ACTIVE, 'Active'),
        (STATUS_CANCELLED, 'Cancelled'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='seat_reservations')
    seat = models.ForeignKey(Seat, on_delete=models.CASCADE, related_name='reservations')
    reservation_date = models.DateField()
    start_time = models.TimeField()
    duration_minutes = models.PositiveSmallIntegerField(default=30)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_ACTIVE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-reservation_date', '-start_time']

    def __str__(self):
        return f'{self.user} reserved {self.seat} on {self.reservation_date}'

    @property
    def starts_at(self):
        return timezone.make_aware(
            datetime.combine(self.reservation_date, self.start_time),
            timezone.get_current_timezone(),
        )

    @property
    def ends_at(self):
        return self.starts_at + timedelta(minutes=self.duration_minutes)

    @property
    def is_current(self):
        now = timezone.now()
        return self.status == self.STATUS_ACTIVE and self.starts_at <= now < self.ends_at
