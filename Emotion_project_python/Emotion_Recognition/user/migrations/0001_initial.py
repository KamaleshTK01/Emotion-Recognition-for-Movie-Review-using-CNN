# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2022-05-06 08:33
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='RecognitionModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=300)),
                ('path', models.FileField(upload_to='')),
                ('result', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='RegisterModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('firstname', models.CharField(max_length=300)),
                ('lastname', models.CharField(max_length=200)),
                ('password', models.CharField(max_length=200)),
                ('repassword', models.CharField(max_length=200)),
                ('email', models.EmailField(max_length=400)),
            ],
        ),
    ]
