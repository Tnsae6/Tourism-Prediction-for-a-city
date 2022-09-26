# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib import admin

from apps.home.models import Hotels, PriceTags, review, upcomingevent

# Register your models here.
admin.site.register(review)
admin.site.register(PriceTags)
admin.site.register(upcomingevent)
admin.site.register(Hotels)