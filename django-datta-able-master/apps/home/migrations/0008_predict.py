# Generated by Django 3.2 on 2022-05-29 10:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0007_hotels_pricetags_upcomingevent'),
    ]

    operations = [
        migrations.CreateModel(
            name='Predict',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('travel_with', models.CharField(choices=[('Friends/Relatives', 'Friends/Relatives'), ('Children', 'Children'), ('Spouse', 'Spouse'), ('Spouse and Children', 'Spouse and Children'), ('Alone', 'Alone'), ('other', 'other')], max_length=50)),
                ('purpose', models.CharField(choices=[('Meetings and Conference', 'meeting and conference'), ('Business', 'Business'), ('Scientific and Academic', 'Scientific and Academic'), ('Volunteering', 'volunteering'), ('Visiting Friends and Relatives', 'Visiting Friends and Relatives'), ('Leasure and Holidays', 'Leasure and Holidays'), ('other', 'other')], max_length=50)),
                ('main_activity', models.CharField(choices=[('Bird watching', 'Bird Watching'), ('Diving and Sport Fishing', 'Fishing'), ('Beach tourism', 'Lake'), ('Business', 'business'), ('Wildlife tourism', 'Wildlife'), ('Cultural tourism', 'Culture'), ('Mountain climbing', 'Hiking'), ('Confernce tourism', 'Conference'), ('Hunting tourism', 'Legal Hunting'), ('others', 'other')], max_length=50)),
                ('infor_source', models.CharField(choices=[('Ethiopia Mission Abroad', 'Ethiopia Mission Abroad'), ('"Friends, Relatives"', 'Friends and Relatives'), ('Inflight megazines', 'Inflight Megazines'), ('"Newspaper, megazine, brochures"', '"Newspaper, megazine, brochures"'), ('others', 'other'), ('"Radio, TV, Web"', 'Media'), ('Trade fair', 'Trade fair'), ('"Travel, agent, tour operator"', '"Travel, agent, tour operator"')], max_length=50)),
                ('tour_arrangment', models.CharField(choices=[('Independent', 'Idependent'), ('Package Tour', 'Package Tour')], max_length=50)),
                ('age_group', models.CharField(choices=[('1-24', 'less than 24'), ('25-44', '25 - 44'), ('45-64', '45-64'), ('65+', '65 or above')], max_length=50)),
            ],
        ),
    ]
