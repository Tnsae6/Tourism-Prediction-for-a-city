from django import forms
from .models import Predict, review

class Reviewform(forms.ModelForm):
    class Meta:
        model = review
        fields = '__all__'
        # fields = ['Id','country', 'age_group','travel_with']

class Predictionform(forms.ModelForm):
    class Meta:
        model = Predict
        fields = '__all__'
        