from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect, render

from .forms import ProfileUpdateForm, RememberMeAuthenticationForm, StudentRegistrationForm, UserUpdateForm
from circulation_app.services import get_my_loans_summary
from library.models import Loan


def architecture(request):
    context = {
        'app_title': 'Accounts App',
        'owner': 'Team Member 1',
        'purpose': 'Student registration, login experience, profile data, and session policies.',
        'modules': [
            'models.py: future profile model',
            'forms.py: registration/profile forms',
            'services.py: account workflows',
            'selectors.py: account queries',
        ],
    }
    return render(request, 'accounts_app/architecture.html', context)


class RememberMeLoginView(LoginView):
    template_name = 'registration/login.html'
    authentication_form = RememberMeAuthenticationForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')
        if remember_me:
            self.request.session.set_expiry(60 * 60 * 24 * 14)
        else:
            self.request.session.set_expiry(0)
        return super().form_valid(form)


def register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = StudentRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Your student account is ready.')
            return redirect('dashboard')
    else:
        form = StudentRegistrationForm()

    return render(request, 'registration/register.html', {'form': form})


@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    summary = get_my_loans_summary(request.user)
    context = {
        **summary,
        'u_form': u_form,
        'p_form': p_form,
    }
    return render(request, 'library/profile.html', context)
