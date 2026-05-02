from datetime import timedelta

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from library.models import Book, Loan


class CirculationFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='student', password='StrongPass123')
        self.book = Book.objects.create(
            title='Django for Beginners',
            author='William S. Vincent',
            isbn='9781735467207',
            category='Web Development',
            total_copies=1,
            available_copies=1,
        )

    def test_student_can_checkout_available_book(self):
        self.client.login(username='student', password='StrongPass123')

        response = self.client.post(reverse('checkout_book'), {'isbn': self.book.isbn})

        self.assertRedirects(response, reverse('my_loans'))
        self.book.refresh_from_db()
        self.assertEqual(self.book.available_copies, 0)
        self.assertTrue(Loan.objects.filter(user=self.user, book=self.book, returned_at__isnull=True).exists())

    def test_student_can_return_active_loan(self):
        self.client.login(username='student', password='StrongPass123')
        loan = Loan.objects.create(user=self.user, book=self.book)
        self.book.available_copies = 0
        self.book.save()

        response = self.client.post(reverse('checkin_book', args=[loan.id]))

        self.assertRedirects(response, reverse('my_loans'))
        loan.refresh_from_db()
        self.book.refresh_from_db()
        self.assertIsNotNone(loan.returned_at)
        self.assertEqual(self.book.available_copies, 1)

    def test_checkout_requires_login(self):
        response = self.client.post(reverse('checkout_book'), {'isbn': self.book.isbn})

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response['Location'])
