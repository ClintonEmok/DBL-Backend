from django.http import HttpResponse
from django.shortcuts import render






def homepage(request):
    # return HttpResponse('Welcome to our page')
    return render(request, "index.html")


def about(request):
    # return HttpResponse('Hello World')
    return render(request, "about.html")


def plot(request):
    return render(request, "plot.html")
