from django.shortcuts import render
from books.models import Book


def search_view(request):
    query = request.GET.get('q', '').strip()
    genre = request.GET.get('genre', '').strip()
    available = request.GET.get('available', '')

    books = Book.objects.all()

    if query:
        books = books.filter(title__icontains=query) | books.filter(author__icontains=query)

    if genre:
        books = books.filter(genre__icontains=genre)

    if available:
        books = books.filter(available_copies__gt=0)

    genres = Book.objects.values_list('genre', flat=True).distinct()

    context = {
        'books': books,
        'query': query,
        'genre': genre,
        'available': available,
        'genres': genres,
    }
    return render(request, 'search/results.html', context)