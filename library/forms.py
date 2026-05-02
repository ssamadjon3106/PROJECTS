from django import forms


class BaseSearchForm(forms.Form):
    """A reusable base search form for different apps."""
    q = forms.CharField(
        required=False, 
        widget=forms.TextInput(attrs={'placeholder': 'Search...', 'class': 'search-input'})
    )
