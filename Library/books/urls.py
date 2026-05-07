from django.urls import path
from . import views

app_name = 'books'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('available/', views.available_books_view, name='available_books'),
    path('<int:pk>/', views.detail_view, name='detail'),
    path('<int:pk>/borrow/', views.borrow_view, name='borrow'),
    path('<int:pk>/return/', views.return_view, name='return'),
    path('my/', views.my_books_view, name='my_books'),
]
