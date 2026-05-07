from django.urls import path
from . import views

app_name = 'seats'

urlpatterns = [
    path('', views.floor_map_view, name='floor_map'),
    path('<int:seat_id>/reserve/', views.reserve_view, name='reserve'),
    path('<int:pk>/cancel/', views.cancel_view, name='cancel'),
    path('my/', views.my_reservations_view, name='my_reservations'),
]