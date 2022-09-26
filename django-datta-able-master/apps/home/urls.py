# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views
from apps.home.tests import predict

urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    path('fillform/', views.review, name='review' ),
    path('predict/', predict, name='predict'),
    path('csvtable/', views.table, name="csvtable"),
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
