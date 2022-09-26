# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.shortcuts import render

from apps.home.forms import Reviewform, Predictionform
from .models import Hotels, PriceTags, review as Review, upcomingevent, Predict
# from app.home import plot
import csv, pickle
import io
from django.shortcuts import render
from django.contrib import messages
# Create your views here.
# one parameter named request


@login_required(login_url="/login/")
def table(request):
    # declaring template
    template = "home/csv.html"
    
# prompt == a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be numbers, Id, country, age_group, travel_with, total_famle, total_male, purpose, main_activity, info_source, tour_arrangment, package_transport_int, package_accomodation, package_food, package_transport_tz, package_sight_seeing, package_guided_tour, package_insurance, night_Arba_minch, night_Gamo_Gofa, payment_mode, first_trip_tz, most_impressing, total_cost' ,
        
    }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    # let's check if it == a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS FILE IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')


        # setup a stream which == when we loop through each line we are able to handle a data in a stream
    io_string = io.StringIO(data_set)
    next(io_string)
    if csv_file.name.endswith('.csv'):
        for column in csv.reader(io_string, delimiter=',', quotechar='"'):
            _, created = Review.objects.update_or_create(
                country=column[1],
                age_group=column[2],
                travel_with=column[3],
                total_female=column[4],
                total_male=column[5],
                purpose = column[6],
                main_activity = column[7],
                info_source = column[8],
                tour_arrangement=column[9],
                package_transport_int = column[10],
                package_accomodation = column[11],
                package_food = column[12],
                package_transport_tz = column[13],
                package_sightseeing = column[14],
                package_guided_tour = column[15],
                package_insurance = column[16],
                night_Arba_minch = column[17],
                night_Gamo_Gofa = column[18],
                payment_mode=column[19],
                first_trip_tz = column[20],
                most_impressing=column[21],
                total_cost = column[22],
            )
        
    return render(request, template, prompt)


@login_required(login_url="/login/")
def index(request):
    review_view = Review.objects.all().order_by('-id')[:6]
    reviews = Review.objects.all()[0:20]
    review_alone = Review.objects.filter(travel_with='Alone').values()[0:20]
    review_not_alone = Review.objects.exclude(travel_with='Alone').values()[0:20]
    PriceTags_view = PriceTags.objects.last()
    upcomingevent_view = upcomingevent.objects.last()
    Hotels_view = Hotels.objects.last()
    context = {'segment': 'index', 'review':review_view,'price':PriceTags_view,
                'upcoming':upcomingevent_view, 'hotels':Hotels_view,
                'alone': review_alone, 'reviews': reviews, 'with': review_not_alone}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))
    

@login_required(login_url="/login/")
def review(request):
    
    if request.method == "POST":
        form = Reviewform(request.POST)
        
        if form.is_valid():
            form.save()
    else:
        form = Reviewform()   
    
    return render(request, "home/form.html", {'form':form})


@login_required(login_url="/login/")
def predict(request):
    if request.method == "POST":
        form = Predictionform(request.POST)

        if form.is_valid():
            form.save()

    filename = 'predict.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    test_x = ['is_african']
    test_y = Predict.objects.last()
    predict = loaded_model.predict(test_x, test_y)
    
    return render(request, "home/predictionform.html", {'actual': predict})



@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]
        context['reviewform']= Reviewform()

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
