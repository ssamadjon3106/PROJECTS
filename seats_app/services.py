from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, TypedDict

from django.db import transaction
from django.utils import timezone

from library.models import Seat, SeatReservation


def reservation_label(seat: Seat, start_time) -> str:
    return f'{seat.label} at {start_time:%H:%M}'


def slot_bounds(reservation_date, start_time, duration_minutes: int):
    starts_at = timezone.make_aware(
        datetime.combine(reservation_date, start_time),
        timezone.get_current_timezone(),
    )
    return starts_at, starts_at + timedelta(minutes=duration_minutes)


def reservation_overlaps(reservation: SeatReservation, starts_at, ends_at) -> bool:
    return reservation.starts_at < ends_at and starts_at < reservation.ends_at


def _build_seat_map_layout(
    *,
    entrance_seats: list[Seat],
    reservations_by_seat_id: dict[int, SeatReservation],
    is_admin: bool,
) -> list[dict[str, Any]]:
    seats: list[dict[str, Any]] = []

    for index, seat in enumerate(entrance_seats):
        reservation = reservations_by_seat_id.get(seat.id)

        status: Literal['available', 'occupied', 'reserved'] = 'available'
        reserved_by = ''
        if not seat.is_active:
            status = 'occupied'
        elif reservation:
            status = 'occupied' if reservation.is_current else 'reserved'
            reserved_by = reservation.user.username if is_admin else ''

        block_index = index % 8
        section = 'A' if index < 8 else 'B'
        seats.append(
            {
                'id': seat.id,
                'label': f'{section}{block_index + 1}',
                'zone': seat.zone,
                'layout_area': 'left' if index < 8 else 'right',
                'layout_row': (block_index % 4) + 1,
                'layout_column': (block_index // 4) + 1,
                'status': status,
                'reserved_by': reserved_by,
            }
        )

    return seats


def get_seat_map_context(
    *,
    user,
    is_admin: bool,
    reservation_date,
    start_time,
    duration_minutes: int,
    entrance_seats: list[Seat],
) -> dict[str, Any]:
    starts_at, ends_at = slot_bounds(reservation_date, start_time, duration_minutes)

    reservations = SeatReservation.objects.select_related('seat', 'user').filter(
        reservation_date=reservation_date,
        status=SeatReservation.STATUS_ACTIVE,
    )

    overlapping = [reservation for reservation in reservations if reservation_overlaps(reservation, starts_at, ends_at)]
    reservations_by_seat_id = {reservation.seat_id: reservation for reservation in overlapping}

    seats = _build_seat_map_layout(
        entrance_seats=entrance_seats,
        reservations_by_seat_id=reservations_by_seat_id,
        is_admin=is_admin,
    )

    user_reservations = (
        SeatReservation.objects.select_related('seat')
        .filter(user=user, status=SeatReservation.STATUS_ACTIVE, reservation_date__gte=timezone.localdate())[:4]
    )

    return {
        'seats': seats,
        'free_count': sum(seat['status'] == 'available' for seat in seats),
        'occupied_count': sum(seat['status'] == 'occupied' for seat in seats),
        'reserved_count': sum(seat['status'] == 'reserved' for seat in seats),
        'user_reservations': user_reservations,
    }


class ReserveSeatResult(TypedDict):
    ok: bool
    error: str | None
    reservation: SeatReservation | None


def reserve_online_seat(
    *,
    user,
    seat: Seat,
    online_seat_ids_list: list[int],
    reservation_date,
    start_time,
    duration_minutes: int,
) -> ReserveSeatResult:
    if seat.id not in online_seat_ids_list:
        return {'ok': False, 'error': 'not_online', 'reservation': None}

    starts_at, ends_at = slot_bounds(reservation_date, start_time, duration_minutes)
    if ends_at <= timezone.now():
        return {'ok': False, 'error': 'time_in_past', 'reservation': None}

    with transaction.atomic():
        reservations = (
            SeatReservation.objects.select_for_update()
            .select_related('seat')
            .filter(reservation_date=reservation_date, status=SeatReservation.STATUS_ACTIVE)
        )

        seat_conflict = any(
            reservation.seat_id == seat.id and reservation_overlaps(reservation, starts_at, ends_at)
            for reservation in reservations
        )
        if seat_conflict:
            return {'ok': False, 'error': 'seat_conflict', 'reservation': None}

        user_conflict = any(
            reservation.user_id == user.id and reservation_overlaps(reservation, starts_at, ends_at)
            for reservation in reservations
        )
        if user_conflict:
            return {'ok': False, 'error': 'user_conflict', 'reservation': None}

        reservation = SeatReservation.objects.create(
            user=user,
            seat=seat,
            reservation_date=reservation_date,
            start_time=start_time,
            duration_minutes=duration_minutes,
        )

    return {'ok': True, 'error': None, 'reservation': reservation}
