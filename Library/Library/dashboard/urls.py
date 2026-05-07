from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('overdue/', views.overdue_view, name='overdue'),
    path('seats/', views.seats_view, name='seats'),
]