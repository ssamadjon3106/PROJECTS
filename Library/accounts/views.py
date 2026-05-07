from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegisterForm, LoginForm


def register_view(request):
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect('books:home')
    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    form = LoginForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        login(request, form.get_user())
        return redirect('books:home')
    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('accounts:login')


@login_required
def profile_view(request):
    return render(request, 'accounts/profile.html')

@login_required
def switch_role_view(request):
    """Toggle the current user between student and librarian.
    Handy for testing the dashboard without touching the shell or admin."""
    if request.method == 'POST':
        u = request.user
        u.role = u.STUDENT if u.role == u.LIBRARIAN else u.LIBRARIAN
        u.save(update_fields=['role'])
        messages.success(request, f'Switched to {u.get_role_display()} mode.')
    return redirect('accounts:profile')