from django.contrib import admin

from library.models import Seat, SeatReservation


@admin.register(Seat)
class SeatAdmin(admin.ModelAdmin):
    list_display = ('label', 'zone', 'row', 'column', 'is_active')
    list_filter = ('zone', 'is_active')
    search_fields = ('label', 'zone')


@admin.register(SeatReservation)
class SeatReservationAdmin(admin.ModelAdmin):
    list_display = ('user', 'seat', 'reservation_date', 'start_time', 'duration_minutes', 'status')
    list_filter = ('reservation_date', 'status', 'seat__zone')
    search_fields = ('user__username', 'seat__label', 'seat__zone')
    autocomplete_fields = ('user', 'seat')
