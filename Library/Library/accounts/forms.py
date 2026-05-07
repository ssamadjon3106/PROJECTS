from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import LibraryUser


class RegisterForm(UserCreationForm):
    class Meta:
        model = LibraryUser
        fields = ['username', 'email', 'password1', 'password2']


class LoginForm(AuthenticationForm):
    pass