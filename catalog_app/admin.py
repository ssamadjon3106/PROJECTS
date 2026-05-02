from django.contrib import admin

from library.models import Book


@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'isbn', 'category', 'available_copies', 'total_copies')
    list_filter = ('category',)
    search_fields = ('title', 'author', 'isbn')
