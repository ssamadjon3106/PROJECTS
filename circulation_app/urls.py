from django.urls import path

from . import views

app_name = 'circulation_app'

urlpatterns = [
    path('', views.architecture, name='architecture'),
]
