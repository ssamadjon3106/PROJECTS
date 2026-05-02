from django import forms


class CatalogSearchForm(forms.Form):
    query = forms.CharField(required=False, max_length=180)
    category = forms.CharField(required=False, max_length=80)
