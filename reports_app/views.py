from django.shortcuts import render
from django.contrib.auth.decorators import user_passes_test

from accounts_app.selectors import get_active_students
from catalog_app.selectors import books_copy_stats
from circulation_app.selectors import active_loans
from seats_app.selectors import active_seats

def architecture(request):
    context = {
        'app_title': 'Reports App',
        'owner': 'Anvar',
        'purpose': 'Project-wide analytics, system health, and dashboard reporting.',
        'modules': [
            'views.py: cross-app data aggregation',
            'urls.py: analytics routing',
            'selectors.py: future specialized report queries',
        ],
    }
    return render(request, 'reports_app/architecture.html', context)

@user_passes_test(lambda u: u.is_staff)
def report_dashboard(request):
    """
    Aggregates library-wide metrics using selectors from all modules.
    Access is restricted to staff members.
    """
    active_student_count = get_active_students().count()
    loan_count = active_loans().count()
    seat_count = active_seats().count()
    inventory_stats = books_copy_stats()

    context = {
        'total_active_students': active_student_count,
        'current_loans_count': loan_count,
        'available_seats_count': seat_count,
        'book_stats': inventory_stats,
    }
    return render(request, 'reports_app/dashboard.html', context)