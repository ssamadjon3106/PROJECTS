from django.db import models

# Create your models here.
from django.db import models
from django.conf import settings
from django.utils import timezone


class Seat(models.Model):
    QUIET = 'quiet'
    GROUP = 'group'
    COMPUTER = 'computer'
    STUDY = 'study'
    LOUNGE = 'lounge'

    ZONE_CHOICES = [
        (QUIET, 'Quiet Zone'),
        (GROUP, 'Group Zone'),
        (COMPUTER, 'Computer Zone'),
        (STUDY, 'Study Zone'),
        (LOUNGE, 'Lounge Zone'),
    ]

    seat_number = models.CharField(max_length=10, unique=True)
    zone = models.CharField(max_length=20, choices=ZONE_CHOICES, default=QUIET)
    pos_x = models.FloatField(default=0)
    pos_y = models.FloatField(default=0)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f'Seat {self.seat_number} ({self.get_zone_display()})'


class Reservation(models.Model):
    ACTIVE = 'active'
    CANCELLED = 'cancelled'
    STATUS_CHOICES = [
        (ACTIVE, 'Active'),
        (CANCELLED, 'Cancelled'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='reservations')
    seat = models.ForeignKey(Seat, on_delete=models.CASCADE, related_name='reservations')
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=ACTIVE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} — {self.seat} ({self.start_time:%Y-%m-%d %H:%M})'
