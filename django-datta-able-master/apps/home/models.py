# -*- encoding, utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class review(models.Model):
    countrieschoice1 = ['ETHIOPIA', 'SWIZERLAND', 'UNITED KINGDOM', 'CHINA', 'SOUTH AFRICA',
                       'UNITED STATES OF AMERICA', 'NIGERIA', 'INDIA', 'BRAZIL', 'CANADA',
                       'MALT', 'MOZAMBIQUE', 'RWANDA', 'AUSTRIA', 'MYANMAR', 'GERMANY',
                       'KENYA', 'ALGERIA', 'IRELAND', 'DENMARK', 'SPAIN', 'FRANCE',
                       'ITALY', 'EGYPT', 'QATAR', 'MALAWI', 'JAPAN', 'SWEDEN',
                       'NETHERLANDS', 'UAE', 'UGANDA', 'AUSTRALIA', 'YEMEN',
                       'NEW ZEALAND', 'BELGIUM', 'NORWAY', 'ZIMBABWE', 'ZAMBIA', 'CONGO',
                       'BURGARIA', 'PAKISTAN', 'GREECE', 'MAURITIUS', 'DRC', 'OMAN',
                       'PORTUGAL', 'KOREA', 'SWAZILAND', 'TUNISIA', 'KUWAIT', 'DOMINICA',
                       'ISRAEL', 'FINLAND', 'CZECH REPUBLIC', 'UKRAIN', 
                       'BURUNDI', 'SCOTLAND', 'RUSSIA', 'GHANA', 'NIGER', 'MALAYSIA',
                       'COLOMBIA', 'LUXEMBOURG', 'NEPAL', 'POLAND', 'SINGAPORE',
                       'LITHUANIA', 'HUNGARY', 'INDONESIA', 'TURKEY', 'TRINIDAD TOBACCO',
                       'IRAQ', 'SLOVENIA', 'UNITED ARAB EMIRATES', 'COMORO', 'SRI LANKA',
                       'IRAN', 'MONTENEGRO', 'ANGOLA', 'LEBANON', 'SLOVAKIA', 'ROMANIA',
                       'MEXICO', 'LATVIA', 'CROATIA', 'CAPE VERDE', 'SUDAN', 'COSTARICA',
                       'CHILE', 'NAMIBIA', 'TAIWAN', 'SERBIA', 'LESOTHO', 'GEORGIA',
                       'PHILIPINES', 'IVORY COAST', 'MADAGASCAR', 'DJIBOUT', 'CYPRUS',
                       'ARGENTINA', 'URUGUAY', 'MORROCO', 'THAILAND', 'BERMUDA',
                       'ESTONIA', 'BOTSWANA', 'BULGARIA', 'BANGLADESH', 'HAITI',
                       'VIETNAM', 'BOSNIA', 'LIBERIA', 'PERU', 'JAMAICA', 'MACEDONIA',
                       'GUINEA', 'SOMALI', 'SAUD ARABIA']
    countrieschoice = []
    i = 0
    while i < len(countrieschoice1):

        j = countrieschoice1[i]
        
        countrieschoice.append((j,j))
        i+=1

    
    agechoices=[('1-24','less than 24'),('25-44','25 - 44'), ('45-64','45-64'), ('65+','65 or above') ]
    
    boolchoices=[('Yes','Yes'), ('No','No')]

    travel_with_choice=[('Friends/Relatives','Friends/Relatives'), ('Children','Children'), ('Spouse','Spouse'), ('Spouse and Children','Spouse and Children'), ('Alone','Alone'), ('other','other')]
    
    purposechoice=[('Meetings and Conference','meeting and conference'), ('Business','Business'), 
                   ('Scientific and Academic', 'Scientific and Academic'), ('Volunteering', 'volunteering'),
                    ('Visiting Friends and Relatives', 'Visiting Friends and Relatives'),
                    ('Leasure and Holidays', 'Leasure and Holidays'),('other', 'other'), ]
    
    activity_choice = [('Bird watching','Bird Watching'), ('Diving and Sport Fishing','Fishing'), ('Beach tourism', 'Lake'),('Business','business'),
                        ('Wildlife tourism','Wildlife'),('Cultural tourism', 'Culture'), ('Mountain climbing','Hiking'),
                        ('Confernce tourism','Conference'), ('Hunting tourism','Legal Hunting'),('others','other')]
    
    infor_source_choice = [('Ethiopia Mission Abroad', 'Ethiopia Mission Abroad'), ('"Friends, Relatives"', 'Friends and Relatives'),
                           ('Inflight megazines', 'Inflight Megazines'), ('"Newspaper, megazine, brochures"', '"Newspaper, megazine, brochures"'),
                            ('others', 'other'), ('"Radio, TV, Web"', 'Media'), ('Trade fair', 'Trade fair'), ('"Travel, agent, tour operator"', '"Travel, agent, tour operator"')]
    
    tour_arrangment_choice = [('Independent','Idependent'),('Package Tour','Package Tour')]

    payment_mode_choice = [('Cash','Cash'), ('Credit Card','Credit Card'), ('Travellers Cheque','Cheque'), ('Other','Other')]
    
    impression_choice =[('Wildlife','Best Wildlife'), ('Excellent Experince', 'Excellent Experience'), ('Friendly People','Friendly People'), ('Good Service','Good Service'),
                         ('No comments','Disappointed'), ('Satisfies and Hope Come Back','Satisfied and Hope to come back'), ('Wonderful Country, Landscape, Nature','Wonderful Country, Landscape, Nature')]
    
    # ID= models.ForeignKey(to=User, on_delete=models.CASCADE)
    country = models.CharField(max_length=50, choices=countrieschoice)
    age_group = models.CharField(max_length=50,  choices=agechoices)
    travel_with = models.CharField(max_length=50, blank = True, null= True, choices=travel_with_choice)
    total_female = models.FloatField()
    total_male = models.FloatField()
    purpose = models.CharField(max_length=50,  choices=purposechoice)
    main_activity = models.CharField(max_length=50, choices=activity_choice)
    info_source = models.CharField(max_length=50, choices=infor_source_choice)
    tour_arrangement = models.CharField(max_length=50, choices=tour_arrangment_choice)
    package_transport_int = models.CharField(max_length=50, choices=boolchoices)
    package_accomodation = models.CharField(max_length=50, choices=boolchoices)
    package_food = models.CharField(max_length=50, choices=boolchoices)
    package_transport_tz = models.CharField(max_length=50, choices=boolchoices)
    package_sightseeing = models.CharField(max_length=50, choices=boolchoices)
    package_guided_tour = models.CharField(max_length=50, choices=boolchoices)
    package_insurance = models.CharField(max_length=50, choices=boolchoices)
    night_Arba_minch = models.FloatField()
    night_Gamo_Gofa = models.FloatField()
    payment_mode = models.CharField(max_length=50, choices=payment_mode_choice)
    first_trip_tz = models.CharField(max_length=50, choices=boolchoices)
    most_impressing = models.CharField(
        max_length=50, blank=True, null=True,  choices=impression_choice)
    total_cost = models.FloatField()
    
    def __str__(self):
        return self.country 


class PriceTags(models.Model):
    average_room_price = models.IntegerField()
    discount_percent = models.IntegerField()
    average_twins_room = models.IntegerField()
    discount_twins_percent = models.IntegerField()
    full_package_cost = models.IntegerField()
    discount_full_percent = models.IntegerField()
    

class upcomingevent(models.Model):
    upcoming_event = models.CharField(max_length=200)
    event_size = models.IntegerField()
    event_date = models.DateField()
    def __str__(self):
        return self.upcoming_event

class Hotels(models.Model):
    available_rooms = models.IntegerField()
    available_twins_room = models.IntegerField()
    

class Predict(models.Model):
    travel_with_choice = [('Friends/Relatives', 'Friends/Relatives'), ('Children', 'Children'), ('Spouse','Spouse'),
                            ('Spouse and Children', 'Spouse and Children'), ('Alone', 'Alone'), ('other', 'other')]

    purposechoice = [('Meetings and Conference', 'meeting and conference'), ('Business', 'Business'),
                     ('Scientific and Academic','Scientific and Academic'), ('Volunteering', 'volunteering'),
                     ('Visiting Friends and Relatives','Visiting Friends and Relatives'),
                     ('Leasure and Holidays', 'Leasure and Holidays'), ('other', 'other'), ]

    activity_choice = [('Bird watching', 'Bird Watching'), ('Diving and Sport Fishing', 'Fishing'), ('Beach tourism', 'Lake'), ('Business', 'business'),
                       ('Wildlife tourism', 'Wildlife'), ('Cultural tourism',
                                                          'Culture'), ('Mountain climbing', 'Hiking'),
                       ('Confernce tourism', 'Conference'), ('Hunting tourism', 'Legal Hunting'), ('others', 'other')]

    infor_source_choice = [('Ethiopia Mission Abroad', 'Ethiopia Mission Abroad'), ('"Friends, Relatives"', 'Friends and Relatives'),
                           ('Inflight megazines', 'Inflight Megazines'), (
                               '"Newspaper, megazine, brochures"', '"Newspaper, megazine, brochures"'),
                           ('others', 'other'), ('"Radio, TV, Web"', 'Media'), ('Trade fair', 'Trade fair'), ('"Travel, agent, tour operator"', '"Travel, agent, tour operator"')]

    tour_arrangment_choice = [
        ('Independent', 'Idependent'), ('Package Tour', 'Package Tour')]
    agechoices = [('1-24', 'less than 24'), ('25-44', '25 - 44'),
                  ('45-64', '45-64'), ('65+', '65 or above')]

    travel_with = models.CharField(max_length=50, choices=travel_with_choice)
    purpose = models.CharField(max_length=50,  choices=purposechoice)
    main_activity = models.CharField(max_length=50, choices=activity_choice)
    infor_source = models.CharField(max_length=50, choices=infor_source_choice)
    tour_arrangment = models.CharField(max_length=50, choices=tour_arrangment_choice)
    age_group = models.CharField(max_length=50,  choices=agechoices)
