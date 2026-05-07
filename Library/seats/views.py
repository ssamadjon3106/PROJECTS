from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django import forms
from .models import Seat, Reservation
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import datetime
import json


# ── Availability filter form ──────────────────────────────────────────────────

class SeatFilterForm(forms.Form):
    reservation_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'}),
        required=False,
    )
    start_time = forms.TimeField(
        widget=forms.TimeInput(attrs={'type': 'time'}),
        required=False,
    )
    duration_minutes = forms.ChoiceField(
        choices=[(30, '30 min'), (60, '1 hour'), (90, '1.5 hours'), (120, '2 hours'), (180, '3 hours')],
        required=False,
        initial=60,
    )


# ── Views ─────────────────────────────────────────────────────────────────────

@login_required
def floor_map_view(request):
    # ── JSON endpoints used by the floor_map JS ──────────────────────────
    if request.GET.get('format') == 'json':
        now = timezone.now()

        if request.method == 'POST' and request.GET.get('action') == 'availability':
            # Preview: which seats are unavailable for {bookingDate, startTime, durationMin}
            try:
                payload = json.loads(request.body or b'{}')
                booking_date = payload.get('bookingDate')
                start_time_s = payload.get('startTime')
                duration_min = int(payload.get('durationMin') or 60)
                start_dt = datetime.datetime.strptime(
                    f"{booking_date} {start_time_s}", "%Y-%m-%d %H:%M"
                )
                if timezone.is_naive(start_dt):
                    start_dt = timezone.make_aware(start_dt)
                end_dt = start_dt + datetime.timedelta(minutes=duration_min)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                return JsonResponse({'error': 'Invalid date/time'}, status=400)

            overlapping = Reservation.objects.filter(
                status=Reservation.ACTIVE,
                start_time__lt=end_dt,
                end_time__gt=start_dt,
            ).select_related('seat').values_list('seat__seat_number', flat=True)

            return JsonResponse({'unavailable': list(set(overlapping))})

        # GET: live seat status — list of seats that are occupied right now
        active_now = Reservation.objects.filter(
            status=Reservation.ACTIVE,
            start_time__lte=now,
            end_time__gt=now,
        ).select_related('seat', 'user')

        holds = [{
            'seatId':    r.seat.seat_number,
            'expiresAt': r.end_time.isoformat(),
            'source':    'booking',
            'studentId': r.user.username,
        } for r in active_now]

        return JsonResponse({'seats': holds})

    # ── Normal HTML render ────────────────────────────────────────────────
    form = SeatFilterForm(request.GET or None)

    now = timezone.now()
    reservation_date = now.date()
    start_time = now.time().replace(second=0, microsecond=0)
    duration_minutes = 60

    if form.is_valid():
        reservation_date = form.cleaned_data.get('reservation_date') or reservation_date
        start_time = form.cleaned_data.get('start_time') or start_time
        duration_minutes = int(form.cleaned_data.get('duration_minutes') or 60)

    # Build the window we are checking
    start_dt = datetime.datetime.combine(reservation_date, start_time)
    if timezone.is_naive(start_dt):
        start_dt = timezone.make_aware(start_dt)
    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)

    all_seats = Seat.objects.filter(is_active=True).prefetch_related('reservations__user')

    seats_context = []
    for i, seat in enumerate(all_seats):
        active_reservation = seat.reservations.filter(
            start_time__lt=end_dt,
            end_time__gt=start_dt,
            status=Reservation.ACTIVE,
        ).select_related('user').first()

        taken = active_reservation is not None

        # Distribute seats into left/right blocks (2 cols x 4 rows each)
        if i < 8:
            area = 'left'
            idx  = i
        else:
            area = 'right'
            idx  = i - 8

        row = (idx // 2) + 1
        col = (idx %  2) + 1

        seats_context.append({
            'id':            seat.id,
            'label':         seat.seat_number,
            'zone':          seat.get_zone_display(),
            'status':        'occupied' if taken else 'available',
            'layout_area':   area,
            'layout_row':    row,
            'layout_column': col,
            'reserved_by':   (active_reservation.user.get_full_name()
                              or active_reservation.user.username)
                             if active_reservation else '',
        })

    free_count     = sum(1 for s in seats_context if s['status'] == 'available')
    occupied_count = len(seats_context) - free_count

    # Build seats_json for the floor_map JS (SEAT_LAYOUT). Positions come
    # straight from the DB — they are NOT changed here.
    seats_payload = []
    for seat in all_seats:
        ctx = next((c for c in seats_context if c['id'] == seat.id), None)
        seats_payload.append({
            'id':           seat.id,
            'number':       seat.seat_number,
            'zone_display': seat.get_zone_display(),
            'pos_x':        seat.pos_x,
            'pos_y':        seat.pos_y,
            'available':    (ctx is None) or (ctx['status'] == 'available'),
        })
    seats_json = json.dumps(seats_payload)

    user_reservations = request.user.reservations.filter(
        status=Reservation.ACTIVE,
        end_time__gte=now,
    ).select_related('seat').order_by('start_time')

    return render(request, 'seats/floor_map.html', {
        'form':              form,
        'seats':             seats_context,
        'seats_json':        seats_json,
        'free_count':        free_count,
        'occupied_count':    occupied_count,
        'reserved_count':    occupied_count,
        'reservation_date':  reservation_date,
        'start_time':        start_time,
        'duration_minutes':  duration_minutes,
        'user_reservations': user_reservations,
    })


@login_required
def reserve_view(request, seat_id):
    seat = get_object_or_404(Seat, pk=seat_id, is_active=True)

    if request.method == 'POST':
        if 'duration_minutes' in request.POST:
            # Posted from floor_map side panel
            reservation_date = request.POST.get('reservation_date')
            start_time_str   = request.POST.get('start_time')
            duration_minutes = int(request.POST.get('duration_minutes', 60))
            try:
                start_dt = timezone.make_aware(
                    datetime.datetime.strptime(f"{reservation_date} {start_time_str}", "%Y-%m-%d %H:%M")
                )
                end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)
            except (ValueError, TypeError):
                messages.error(request, 'Invalid date or time. Please try again.')
                return redirect('seats:floor_map')
        else:
            # Posted from reserve.html (datetime-local inputs)
            try:
                start_dt = timezone.make_aware(
                    datetime.datetime.strptime(request.POST.get('start_time', ''), "%Y-%m-%dT%H:%M")
                )
                end_dt = timezone.make_aware(
                    datetime.datetime.strptime(request.POST.get('end_time', ''), "%Y-%m-%dT%H:%M")
                )
            except (ValueError, TypeError):
                messages.error(request, 'Invalid date or time.')
                return redirect('seats:reserve', seat_id=seat_id)

        if end_dt <= start_dt:
            messages.error(request, 'End time must be after start time.')
            return redirect('seats:reserve', seat_id=seat_id)

        overlap = Reservation.objects.filter(
            seat=seat,
            status=Reservation.ACTIVE,
            start_time__lt=end_dt,
            end_time__gt=start_dt,
        ).exists()

        if overlap:
            messages.error(request, 'This seat is already reserved for that time.')
            return redirect('seats:floor_map')

        Reservation.objects.create(
            user=request.user,
            seat=seat,
            start_time=start_dt,
            end_time=end_dt,
        )
        request.user.points += 5
        request.user.save()
        request.session['last_reserved_seat'] = seat.seat_number
        messages.success(request, f'Seat {seat.seat_number} reserved!')
        return redirect('seats:my_reservations')

    return render(request, 'seats/reserve.html', {'seat': seat})


@login_required
def cancel_view(request, pk):
    reservation = get_object_or_404(Reservation, pk=pk, user=request.user)
    reservation.status = Reservation.CANCELLED
    reservation.save()
    messages.success(request, 'Reservation cancelled.')
    return redirect('seats:my_reservations')


@login_required
def my_reservations_view(request):
    reservations = request.user.reservations.select_related('seat').order_by('-created_at')
    return render(request, 'seats/my_reservations.html', {
        'reservations': reservations,
        'now': timezone.now(),
    })