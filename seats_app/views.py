from urllib.parse import urlencode

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone

from library.models import Seat, SeatReservation
from .forms import SeatReservationForm
from .selectors import online_seat_ids
from .services import get_seat_map_context, reserve_online_seat


def architecture(request):
    context = {
        'app_title': 'Seats App',
        'owner': 'Team Member 4',
        'purpose': 'Online seat reservations, map UI, availability checks, and reservation passes.',
        'modules': [
            'models.py: future seat/reservation models',
            'forms.py: date/time reservation forms',
            'services.py: availability and booking logic',
            'selectors.py: seat map queries',
        ],
    }
    return render(request, 'seats_app/architecture.html', context)


@login_required
def seat_reservations(request):
    form = SeatReservationForm(request.GET or None)
    if form.is_valid():
        reservation_date = form.cleaned_data['reservation_date']
        start_time = form.cleaned_data['start_time']
        duration_minutes = form.cleaned_data['duration_minutes']
    else:
        reservation_date = timezone.localdate()
        start_time = timezone.localtime().replace(second=0, microsecond=0).time()
        duration_minutes = 30
        form = SeatReservationForm(
            initial={
                'reservation_date': reservation_date,
                'start_time': start_time,
                'duration_minutes': duration_minutes,
            }
        )

    is_admin = bool(request.user.is_staff)
    entrance_seats = Seat.objects.filter(id__in=online_seat_ids()).order_by('label', 'id')

    seat_map = get_seat_map_context(
        user=request.user,
        is_admin=is_admin,
        reservation_date=reservation_date,
        start_time=start_time,
        duration_minutes=duration_minutes,
        entrance_seats=list(entrance_seats),
    )

    context = {
        'form': form,
        'seats': seat_map['seats'],
        'free_count': seat_map['free_count'],
        'occupied_count': seat_map['occupied_count'],
        'reserved_count': seat_map['reserved_count'],
        'reservation_date': reservation_date,
        'start_time': start_time,
        'duration_minutes': duration_minutes,
        'user_reservations': seat_map['user_reservations'],
    }
    return render(request, 'library/seat_reservations.html', context)


@login_required
def reserve_seat(request):
    if request.method != 'POST':
        return redirect('seat_reservations')

    form = SeatReservationForm(request.POST)
    if not form.is_valid() or not form.cleaned_data.get('seat_id'):
        messages.error(request, 'Choose a valid seat and time slot.')
        return redirect('seat_reservations')

    reservation_date = form.cleaned_data['reservation_date']
    start_time = form.cleaned_data['start_time']
    duration_minutes = form.cleaned_data['duration_minutes']
    seat = get_object_or_404(Seat, id=form.cleaned_data['seat_id'], is_active=True)

    query = urlencode(
        {
            'reservation_date': reservation_date.isoformat(),
            'start_time': start_time.strftime('%H:%M'),
            'duration_minutes': duration_minutes,
        }
    )

    online_ids = online_seat_ids()
    result = reserve_online_seat(
        user=request.user,
        seat=seat,
        online_seat_ids_list=online_ids,
        reservation_date=reservation_date,
        start_time=start_time,
        duration_minutes=duration_minutes,
    )

    if not result['ok']:
        error = result['error']
        if error == 'not_online':
            messages.error(request, 'Online reservations are limited to the blue-zone seats.')
        elif error == 'time_in_past':
            messages.error(request, 'Please choose a future time slot.')
        elif error == 'seat_conflict':
            messages.error(request, f'Seat {seat.label} is already reserved for that time.')
        elif error == 'user_conflict':
            messages.error(request, 'You already have a reservation during that time.')
        else:
            messages.error(request, 'Unable to reserve the seat for that time.')
        return redirect(f'{reverse("seat_reservations")}?{query}')

    reservation = result['reservation']
    request.session['last_reserved_seat'] = seat.label
    messages.success(
        request,
        f'Seat {reservation.seat.label} reserved for {reservation.start_time.strftime("%H:%M")}.',
    )
    return redirect(f'{reverse("seat_reservations")}?{query}')
