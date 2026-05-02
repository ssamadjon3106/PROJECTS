from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from .models import Profile


class StudentRegistrationForm(UserCreationForm):
    first_name = forms.CharField(max_length=150)
    last_name = forms.CharField(max_length=150)
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')


class RememberMeAuthenticationForm(AuthenticationForm):
    remember_me = forms.BooleanField(required=False, initial=True)


class ProfileSearchForm(forms.Form):
    query = forms.CharField(required=False, max_length=120)


class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email')


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ('student_id', 'phone_number')
