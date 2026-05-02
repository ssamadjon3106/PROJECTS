from django import forms


class ReportFilterForm(forms.Form):
    date_from = forms.DateField(required=False)
    date_to = forms.DateField(required=False)
