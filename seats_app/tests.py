from datetime import timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from library.models import Seat, SeatReservation


class SeatReservationFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='student', password='StrongPass123')

    def test_student_can_reserve_available_seat(self):
        seat = Seat.objects.create(label='A1', zone='Entrance', row=1, column=1)
        reservation_date = timezone.localdate() + timedelta(days=1)

        self.client.login(username='student', password='StrongPass123')
        response = self.client.post(
            reverse('reserve_seat'),
            {
                'reservation_date': reservation_date.isoformat(),
                'start_time': '10:00',
                'duration_minutes': '30',
                'seat_id': seat.id,
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            SeatReservation.objects.filter(
                user=self.user,
                seat=seat,
                reservation_date=reservation_date,
            ).exists()
        )

    def test_reserve_seat_conflict_same_seat_overlapping_time(self):
        other_user = User.objects.create_user(username='student2', password='StrongPass123')

        seat = Seat.objects.create(label='A2', zone='Entrance', row=1, column=2)
        reservation_date = timezone.localdate() + timedelta(days=1)

        # First user reserves the seat for the slot.
        self.client.login(username='student', password='StrongPass123')
        first_resp = self.client.post(
            reverse('reserve_seat'),
            {
                'reservation_date': reservation_date.isoformat(),
                'start_time': '10:00',
                'duration_minutes': '30',
                'seat_id': seat.id,
            },
        )
        self.assertEqual(first_resp.status_code, 302)

        # Second user tries to reserve the same seat for an overlapping slot.
        self.client.login(username='student2', password='StrongPass123')
        second_resp = self.client.post(
            reverse('reserve_seat'),
            {
                'reservation_date': reservation_date.isoformat(),
                'start_time': '10:00',
                'duration_minutes': '30',
                'seat_id': seat.id,
            },
        )
        self.assertEqual(second_resp.status_code, 302)

        # Conflict should prevent the second reservation.
        self.assertTrue(
            SeatReservation.objects.filter(user=self.user, seat=seat, reservation_date=reservation_date).exists()
        )
        self.assertFalse(
            SeatReservation.objects.filter(user=other_user, seat=seat, reservation_date=reservation_date).exists()
        )

    def test_seat_map_masks_reserved_by_for_non_admin(self):
        reserved_user = self.user
        other_user = User.objects.create_user(username='student2', password='StrongPass123', is_active=True)

        seat = Seat.objects.create(label='A1', zone='Entrance Left', row=1, column=4)
        reservation_date = timezone.localdate() + timedelta(days=1)
        SeatReservation.objects.create(
            user=reserved_user,
            seat=seat,
            reservation_date=reservation_date,
            start_time='10:00',
            duration_minutes=30,
        )

        self.client.login(username=other_user.username, password='StrongPass123')
        resp = self.client.get(
            reverse('seat_reservations'),
            {
                'reservation_date': reservation_date.isoformat(),
                'start_time': '10:00',
                'duration_minutes': '30',
            },
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.content.decode('utf-8')
        self.assertNotIn(f'data-reserved-by="{reserved_user.username}"', html)
        self.assertNotIn(f'>{reserved_user.username}<', html)

    def test_seat_map_shows_reserved_by_for_admin(self):
        reserved_user = self.user
        admin_user = User.objects.create_user(username='admin1', password='StrongPass123', is_staff=True, is_active=True)

        seat = Seat.objects.create(label='A1', zone='Entrance Left', row=1, column=4)
        reservation_date = timezone.localdate() + timedelta(days=1)
        SeatReservation.objects.create(
            user=reserved_user,
            seat=seat,
            reservation_date=reservation_date,
            start_time='10:00',
            duration_minutes=30,
        )

        self.client.login(username=admin_user.username, password='StrongPass123')
        resp = self.client.get(
            reverse('seat_reservations'),
            {
                'reservation_date': reservation_date.isoformat(),
                'start_time': '10:00',
                'duration_minutes': '30',
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(reserved_user.username, resp.content.decode('utf-8'))

    def test_reserve_seat_rejects_non_entrance_zone(self):
        reservation_seat = Seat.objects.create(label='X1', zone='Library Hall', row=1, column=1, is_active=True)
        reservation_date = timezone.localdate() + timedelta(days=1)

        self.client.login(username=self.user.username, password='StrongPass123')
        resp = self.client.post(
            reverse('reserve_seat'),
            {
                'reservation_date': reservation_date.isoformat(),
                'start_time': '10:00',
                'duration_minutes': '30',
                'seat_id': reservation_seat.id,
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(
            SeatReservation.objects.filter(user=self.user, seat=reservation_seat, reservation_date=reservation_date).exists()
        )
