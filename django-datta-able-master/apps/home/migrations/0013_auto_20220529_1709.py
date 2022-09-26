# Generated by Django 3.2 on 2022-05-29 14:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0012_rename_tour_arrangment_review_tour_arrangement'),
    ]

    operations = [
        migrations.AlterField(
            model_name='review',
            name='age_group',
            field=models.CharField(blank=True, choices=[('1-24', 'less than 24'), ('25-44', '25 - 44'), ('45-64', '45-64'), ('65+', '65 or above')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='country',
            field=models.CharField(blank=True, choices=[('ETHIOPIA', 'ETHIOPIA'), ('SWIZERLAND', 'SWIZERLAND'), ('UNITED KINGDOM', 'UNITED KINGDOM'), ('CHINA', 'CHINA'), ('SOUTH AFRICA', 'SOUTH AFRICA'), ('UNITED STATES OF AMERICA', 'UNITED STATES OF AMERICA'), ('NIGERIA', 'NIGERIA'), ('INDIA', 'INDIA'), ('BRAZIL', 'BRAZIL'), ('CANADA', 'CANADA'), ('MALT', 'MALT'), ('MOZAMBIQUE', 'MOZAMBIQUE'), ('RWANDA', 'RWANDA'), ('AUSTRIA', 'AUSTRIA'), ('MYANMAR', 'MYANMAR'), ('GERMANY', 'GERMANY'), ('KENYA', 'KENYA'), ('ALGERIA', 'ALGERIA'), ('IRELAND', 'IRELAND'), ('DENMARK', 'DENMARK'), ('SPAIN', 'SPAIN'), ('FRANCE', 'FRANCE'), ('ITALY', 'ITALY'), ('EGYPT', 'EGYPT'), ('QATAR', 'QATAR'), ('MALAWI', 'MALAWI'), ('JAPAN', 'JAPAN'), ('SWEDEN', 'SWEDEN'), ('NETHERLANDS', 'NETHERLANDS'), ('UAE', 'UAE'), ('UGANDA', 'UGANDA'), ('AUSTRALIA', 'AUSTRALIA'), ('YEMEN', 'YEMEN'), ('NEW ZEALAND', 'NEW ZEALAND'), ('BELGIUM', 'BELGIUM'), ('NORWAY', 'NORWAY'), ('ZIMBABWE', 'ZIMBABWE'), ('ZAMBIA', 'ZAMBIA'), ('CONGO', 'CONGO'), ('BURGARIA', 'BURGARIA'), ('PAKISTAN', 'PAKISTAN'), ('GREECE', 'GREECE'), ('MAURITIUS', 'MAURITIUS'), ('DRC', 'DRC'), ('OMAN', 'OMAN'), ('PORTUGAL', 'PORTUGAL'), ('KOREA', 'KOREA'), ('SWAZILAND', 'SWAZILAND'), ('TUNISIA', 'TUNISIA'), ('KUWAIT', 'KUWAIT'), ('DOMINICA', 'DOMINICA'), ('ISRAEL', 'ISRAEL'), ('FINLAND', 'FINLAND'), ('CZECH REPUBLIC', 'CZECH REPUBLIC'), ('UKRAIN', 'UKRAIN'), ('BURUNDI', 'BURUNDI'), ('SCOTLAND', 'SCOTLAND'), ('RUSSIA', 'RUSSIA'), ('GHANA', 'GHANA'), ('NIGER', 'NIGER'), ('MALAYSIA', 'MALAYSIA'), ('COLOMBIA', 'COLOMBIA'), ('LUXEMBOURG', 'LUXEMBOURG'), ('NEPAL', 'NEPAL'), ('POLAND', 'POLAND'), ('SINGAPORE', 'SINGAPORE'), ('LITHUANIA', 'LITHUANIA'), ('HUNGARY', 'HUNGARY'), ('INDONESIA', 'INDONESIA'), ('TURKEY', 'TURKEY'), ('TRINIDAD TOBACCO', 'TRINIDAD TOBACCO'), ('IRAQ', 'IRAQ'), ('SLOVENIA', 'SLOVENIA'), ('UNITED ARAB EMIRATES', 'UNITED ARAB EMIRATES'), ('COMORO', 'COMORO'), ('SRI LANKA', 'SRI LANKA'), ('IRAN', 'IRAN'), ('MONTENEGRO', 'MONTENEGRO'), ('ANGOLA', 'ANGOLA'), ('LEBANON', 'LEBANON'), ('SLOVAKIA', 'SLOVAKIA'), ('ROMANIA', 'ROMANIA'), ('MEXICO', 'MEXICO'), ('LATVIA', 'LATVIA'), ('CROATIA', 'CROATIA'), ('CAPE VERDE', 'CAPE VERDE'), ('SUDAN', 'SUDAN'), ('COSTARICA', 'COSTARICA'), ('CHILE', 'CHILE'), ('NAMIBIA', 'NAMIBIA'), ('TAIWAN', 'TAIWAN'), ('SERBIA', 'SERBIA'), ('LESOTHO', 'LESOTHO'), ('GEORGIA', 'GEORGIA'), ('PHILIPINES', 'PHILIPINES'), ('IVORY COAST', 'IVORY COAST'), ('MADAGASCAR', 'MADAGASCAR'), ('DJIBOUT', 'DJIBOUT'), ('CYPRUS', 'CYPRUS'), ('ARGENTINA', 'ARGENTINA'), ('URUGUAY', 'URUGUAY'), ('MORROCO', 'MORROCO'), ('THAILAND', 'THAILAND'), ('BERMUDA', 'BERMUDA'), ('ESTONIA', 'ESTONIA'), ('BOTSWANA', 'BOTSWANA'), ('BULGARIA', 'BULGARIA'), ('BANGLADESH', 'BANGLADESH'), ('HAITI', 'HAITI'), ('VIETNAM', 'VIETNAM'), ('BOSNIA', 'BOSNIA'), ('LIBERIA', 'LIBERIA'), ('PERU', 'PERU'), ('JAMAICA', 'JAMAICA'), ('MACEDONIA', 'MACEDONIA'), ('GUINEA', 'GUINEA'), ('SOMALI', 'SOMALI'), ('SAUD ARABIA', 'SAUD ARABIA')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='first_trip_tz',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='info_source',
            field=models.CharField(blank=True, choices=[('Ethiopia Mission Abroad', 'Ethiopia Mission Abroad'), ('"Friends, Relatives"', 'Friends and Relatives'), ('Inflight megazines', 'Inflight Megazines'), ('"Newspaper, megazine, brochures"', '"Newspaper, megazine, brochures"'), ('others', 'other'), ('"Radio, TV, Web"', 'Media'), ('Trade fair', 'Trade fair'), ('"Travel, agent, tour operator"', '"Travel, agent, tour operator"')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='main_activity',
            field=models.CharField(blank=True, choices=[('Bird watching', 'Bird Watching'), ('Diving and Sport Fishing', 'Fishing'), ('Beach tourism', 'Lake'), ('Business', 'business'), ('Wildlife tourism', 'Wildlife'), ('Cultural tourism', 'Culture'), ('Mountain climbing', 'Hiking'), ('Confernce tourism', 'Conference'), ('Hunting tourism', 'Legal Hunting'), ('others', 'other')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='night_Arba_minch',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='night_Gamo_Gofa',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='package_accomodation',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='package_food',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='package_guided_tour',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='package_insurance',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='package_sightseeing',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='package_transport_int',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='package_transport_tz',
            field=models.CharField(blank=True, choices=[('Yes', 'Yes'), ('No', 'No')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='payment_mode',
            field=models.CharField(blank=True, choices=[('Cash', 'Cash'), ('Credit Card', 'Credit Card'), ('Travellers Cheque', 'Cheque'), ('Other', 'Other')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='purpose',
            field=models.CharField(blank=True, choices=[('Meetings and Conference', 'meeting and conference'), ('Business', 'Business'), ('Scientific and Academic', 'Scientific and Academic'), ('Volunteering', 'volunteering'), ('Visiting Friends and Relatives', 'Visiting Friends and Relatives'), ('Leasure and Holidays', 'Leasure and Holidays'), ('other', 'other')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='total_cost',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='total_female',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='total_male',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='review',
            name='tour_arrangement',
            field=models.CharField(blank=True, choices=[('Independent', 'Idependent'), ('Package Tour', 'Package Tour')], max_length=50, null=True),
        ),
    ]