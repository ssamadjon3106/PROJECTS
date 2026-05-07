from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import LibraryUser


@admin.register(LibraryUser)
class LibraryUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'role', 'points', 'streak_days']
    fieldsets = UserAdmin.fieldsets + (
        ('Library Info', {'fields': ('role', 'points', 'streak_days', 'last_activity_date')}),
    )