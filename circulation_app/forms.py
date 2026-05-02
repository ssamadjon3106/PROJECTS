from django import forms


class CheckoutForm(forms.Form):
    isbn = forms.CharField(
        label='Book ISBN',
        max_length=20,
        widget=forms.TextInput(attrs={'placeholder': 'Scan or type ISBN'}),
    )


class CheckinForm(forms.Form):
    loan_id = forms.IntegerField(widget=forms.HiddenInput)


class LoanLookupForm(forms.Form):
    isbn = forms.CharField(required=False, max_length=20)
    username = forms.CharField(required=False, max_length=150)
