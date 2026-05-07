from django.contrib import admin
from .models import Seat, Reservation


@admin.register(Seat)
class SeatAdmin(admin.ModelAdmin):
    list_display = ['seat_number', 'zone', 'pos_x', 'pos_y', 'is_active']


@admin.register(Reservation)
class ReservationAdmin(admin.ModelAdmin):
    list_display = ['user', 'seat', 'start_time', 'end_time', 'status']