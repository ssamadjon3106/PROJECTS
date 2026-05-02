from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .forms import CheckinForm, CheckoutForm
from .services import checkin_loan_for_user, checkout_book_for_user, get_my_loans_summary


def architecture(request):
    context = {
        'app_title': 'Circulation App',
        'owner': 'Team Member 3',
        'purpose': 'Book checkout, check-in, due dates, overdue rules, and borrowing history.',
        'modules': [
            'models.py: future loan models/policies',
            'forms.py: checkout/checkin forms',
            'services.py: transactional borrowing logic',
            'selectors.py: loan history queries',
        ],
    }
    return render(request, 'circulation_app/architecture.html', context)


@login_required
def checkout_book(request):
    if request.method != 'POST':
        return redirect('dashboard')

    form = CheckoutForm(request.POST)
    if not form.is_valid():
        messages.error(request, 'Please enter a valid ISBN.')
        return redirect('dashboard')

    isbn = form.cleaned_data['isbn'].strip()
    result = checkout_book_for_user(request.user, isbn)

    if not result['ok']:
        error = result['error']
        book = result.get('book')
        if error == 'not_found':
            messages.error(request, 'No book was found with that ISBN.')
            return redirect('dashboard')
        if error == 'unavailable' and book is not None:
            messages.error(request, f'"{book.title}" is currently unavailable.')
            return redirect('dashboard')
        if error == 'already_checked_out' and book is not None:
            messages.info(request, f'You already checked out "{book.title}".')
            return redirect('dashboard')
        messages.error(request, 'Unable to checkout this book.')
        return redirect('dashboard')

    book = result['book']
    request.session['last_checkout_isbn'] = isbn
    messages.success(
        request,
        f'Checked out "{book.title}". Please return it before the due date.',
    )
    return redirect('my_loans')


@login_required
def checkin_book(request, loan_id: int):
    if request.method != 'POST':
        return redirect('my_loans')

    result = checkin_loan_for_user(request.user, loan_id)
    if not result['ok']:
        messages.error(request, 'Loan not found.')
        return redirect('my_loans')

    book = result['book']
    request.session['last_returned_book'] = book.title
    messages.success(request, f'Returned "{book.title}". Thank you.')
    return redirect('my_loans')


@login_required
def my_loans(request):
    summary = get_my_loans_summary(request.user)

    return render(
        request,
        'library/my_loans.html',
        {
            'loans': summary['loans'],
            'active_count': summary['active_count'],
            'returned_count': summary['returned_count'],
            'overdue_count': summary['overdue_count'],
            'checkin_form': CheckinForm(),
        },
    )
