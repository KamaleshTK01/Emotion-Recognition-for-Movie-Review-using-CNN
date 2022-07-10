from django.db import models

# Create your models here.



class RegisterModel(models.Model):
    firstname=models.CharField(max_length=300)
    lastname=models.CharField(max_length=200)
    password=models.CharField(max_length=200)
    repassword=models.CharField(max_length=200)
    email=models.EmailField(max_length=400)

class RecognitionModel(models.Model):
    username=models.CharField(max_length=300)
    path=models.FileField()
    result=models.CharField(max_length=200)
