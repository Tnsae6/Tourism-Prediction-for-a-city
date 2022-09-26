# Generated by Django 3.2 on 2022-05-28 18:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0005_auto_20220528_1705'),
    ]

    operations = [
        migrations.AlterField(
            model_name='review',
            name='infor_source',
            field=models.CharField(choices=[('Ethiopia Mission Abroad', 'Ethiopia Mission Abroad'), ('"Friends, Relatives"', 'Friends and Relatives'), ('Inflight megazines', 'Inflight Megazines'), ('"Newspaper, megazine, brochures"', '"Newspaper, megazine, brochures"'), ('others', 'other'), ('"Radio, TV, Web"', 'Media'), ('Trade fair', 'Trade fair'), ('"Travel, agent, tour operator"', '"Travel, agent, tour operator"')], max_length=50),
        ),
    ]
