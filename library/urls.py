from django.urls import path

from accounts_app.views import profile, register
from catalog_app.views import book_detail, dashboard
from circulation_app.views import checkin_book, checkout_book, my_loans
from seats_app.views import reserve_seat, seat_reservations

urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('register/', register, name='register'),
    path('books/<int:book_id>/', book_detail, name='book_detail'),
    path('checkout/', checkout_book, name='checkout_book'),
    path('loans/', my_loans, name='my_loans'),
    path('loans/<int:loan_id>/return/', checkin_book, name='checkin_book'),
    path('profile/', profile, name='profile'),
    path('seats/', seat_reservations, name='seat_reservations'),
    path('seats/reserve/', reserve_seat, name='reserve_seat'),
]
