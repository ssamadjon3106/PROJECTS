from django.shortcuts import get_object_or_404, render

from circulation_app.forms import CheckoutForm
from library.models import Book

from .selectors import (
    books_copy_stats,
    categories_list,
    get_active_loans_for_user,
    overdue_count_for_user,
    search_books,
    user_has_active_loan,
)


def architecture(request):
    context = {
        'app_title': 'Catalog App',
        'owner': 'Team Member 2',
        'purpose': 'Book inventory, search, categories, imports, and catalog presentation.',
        'modules': [
            'models.py: future book/category models',
            'forms.py: search/import forms',
            'services.py: import and indexing workflows',
            'selectors.py: catalog queries',
        ],
    }
    return render(request, 'catalog_app/architecture.html', context)


def dashboard(request):
    query = request.GET.get('q', '').strip()
    selected_category = request.GET.get('category', '').strip()
    available_only = request.GET.get('available') == '1'

    books = search_books(
        query=query,
        selected_category=selected_category,
        available_only=available_only,
    )

    categories = categories_list()

    stats = books_copy_stats()

    if request.user.is_authenticated:
        active_loans = get_active_loans_for_user(request.user)
        overdue_count = overdue_count_for_user(request.user)
        stats['active_loans'] = active_loans.count()
    else:
        active_loans = []
        overdue_count = 0
        stats['active_loans'] = 0

    context = {
        'books': books,
        'query': query,
        'selected_category': selected_category,
        'available_only': available_only,
        'categories': categories,
        'active_loans': active_loans,
        'overdue_count': overdue_count,
        'stats': stats,
        'checkout_form': CheckoutForm(),
    }
    return render(request, 'library/dashboard.html', context)


def book_detail(request, book_id: int):
    book = get_object_or_404(Book, id=book_id)
    user_has_active_loan_value = user_has_active_loan(user=request.user, book=book)

    return render(
        request,
        'library/book_detail.html',
        {
            'book': book,
            'user_has_active_loan': user_has_active_loan_value,
        },
    )
