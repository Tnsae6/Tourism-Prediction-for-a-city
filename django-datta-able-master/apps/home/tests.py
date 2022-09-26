# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.test import TestCase
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.shortcuts import render

from apps.home.forms import Reviewform, Predictionform
from .models import Hotels, PriceTags, review as Review, upcomingevent, Predict
# from app.home import plot
import csv
import io
from django.shortcuts import render
from django.contrib import messages
# Create your tests here.


@login_required(login_url="/login/")
def predict(request):
    actualprice = 0
    if request.method == "POST":
        form = Predictionform(request.POST)

        if form.is_valid():
            actualprice = 10474
            form.save()
            prediction = Predict.objects.last()
            if prediction.travel_with == 'Friends/Relatives':
                actualprice = actualprice+393
            elif prediction.travel_with == 'Children':
                actualprice = actualprice+14091
            elif prediction.travel_with == 'Spouse':
                actualprice = actualprice+8108
            elif prediction.travel_with == 'Spouse and Children':
                actualprice = actualprice+16931
            else:
                actualprice = actualprice

            if prediction.purpose == 'Meetings and Conference':
                actualprice = actualprice+1078
            elif prediction.purpose == 'Business':
                actualprice = actualprice-1368
            elif prediction.purpose == 'Scientific and Academic':
                actualprice = actualprice-1235
            elif prediction.purpose == 'Volunteering':
                actualprice = actualprice+3090
            elif prediction.purpose == 'Visiting Friends and Relatives':
                actualprice = actualprice-1054
            elif prediction.purpose == 'Leasure and Holidays':
                actualprice = actualprice+4446
            else:
                actualprice = actualprice

            if prediction.age_group == '1-24':
                actualprice = actualprice + 1678
            elif prediction.age_group == '25-44':
                actualprice = actualprice + 2756
            elif prediction.age_group == '45-64':
                actualprice = actualprice + 1954
            else:
                actualprice = actualprice + 547

            if prediction.main_activity == 'Bird watching':
                actualprice = actualprice + 1678
            elif prediction.main_activity == 'Diving and Sport Fishing':
                actualprice = actualprice + 1568
            elif prediction.main_activity == 'Beach tourism':
                actualprice = actualprice + 1589
            elif prediction.main_activity == 'Business':
                actualprice = actualprice + 1678
            elif prediction.main_activity == 'Wildlife tourism':
                actualprice = actualprice + 1047
            elif prediction.main_activity == 'Mountain climbing':
                actualprice = actualprice + 1198
            elif prediction.main_activity == 'Cultural tourism':
                actualprice = actualprice + 1146
            elif prediction.main_activity == 'Confernce tourism':
                actualprice = actualprice + 1782
            elif prediction.main_activity == 'Hunting tourism':
                actualprice = actualprice + 1982

            else:
                actualprice = actualprice

            if prediction.infor_source == 'Ethiopia Mission Abroad':
                actualprice = actualprice + 156
            elif prediction.infor_source == '"Friends, Relatives"':
                actualprice = actualprice + 189

            elif prediction.infor_source == 'Inflight megazines':
                actualprice = actualprice + 211
            elif prediction.infor_source == '"Newspaper, megazine, brochures"':
                actualprice = actualprice + 120
            elif prediction.infor_source == '"Radio, TV, Web"':
                actualprice = actualprice + 192
            elif prediction.infor_source == 'Trade fair':
                actualprice = actualprice + 109
            else:
                actualprice = actualprice + 201
            if prediction.tour_arrangment == 'Independent':
                actualprice = actualprice + 1557
            else:
                actualprice = actualprice + 322

        return render(request, "home/predictionform.html", {'form': form, 'actual': actualprice})

    else:
        form = Predictionform()
        return render(request, "home/predictionform.html", {'form': form})
