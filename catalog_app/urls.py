from django.urls import path

from . import views

app_name = 'catalog_app'

urlpatterns = [
    path('', views.architecture, name='architecture'),
]
