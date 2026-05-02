from django.contrib import admin

from library.models import Loan


@admin.register(Loan)
class LoanAdmin(admin.ModelAdmin):
    list_display = ('user', 'book', 'checked_out_at', 'due_at', 'returned_at')
    list_filter = ('returned_at', 'due_at')
    search_fields = ('user__username', 'book__title', 'book__isbn')
    autocomplete_fields = ('user', 'book')
