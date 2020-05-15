from django.contrib import admin
from django.urls import path
from contact.views import contact, contact_upload

urlpatterns = [
    path('upload-csv/', contact_upload,name='contact_upload'),
]