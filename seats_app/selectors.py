from library.models import Seat


def active_seats():
    return Seat.objects.filter(is_active=True).order_by('zone', 'label')


def online_seat_ids():
    return list(
        Seat.objects.filter(zone__startswith='Entrance')
        .order_by('label', 'id')
        .values_list('id', flat=True)[:16]
    )
