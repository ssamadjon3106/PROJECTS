from django.urls import path
from . import views

app_name = 'reports_app'

urlpatterns = [
    # Architecture demo page
    path('', views.architecture, name='architecture'),
    # Actual analytics dashboard
    path('dashboard/', views.report_dashboard, name='dashboard'),
]