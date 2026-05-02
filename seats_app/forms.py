from django import forms
from django.utils import timezone


class SeatLookupForm(forms.Form):
    date = forms.DateField(required=False)
    zone = forms.CharField(required=False, max_length=60)


class SeatReservationForm(forms.Form):
    DURATION_CHOICES = [
        (30, '30 min'),
        (60, '1 hour'),
        (90, '1.5 hours'),
        (120, '2 hours'),
    ]

    reservation_date = forms.DateField(
        initial=timezone.localdate,
        widget=forms.DateInput(attrs={'type': 'date'}),
    )
    start_time = forms.TimeField(
        initial=lambda: timezone.localtime().replace(second=0, microsecond=0).time(),
        widget=forms.TimeInput(attrs={'type': 'time'}),
    )
    duration_minutes = forms.TypedChoiceField(
        choices=DURATION_CHOICES,
        coerce=int,
        initial=30,
    )
    seat_id = forms.IntegerField(required=False, widget=forms.HiddenInput)
