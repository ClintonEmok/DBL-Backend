from django.shortcuts import render
from django.views.generic import TemplateView

class Home(TemplateView):
    template_name = 'home.html'

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['data']
        print(uploaded_file.name)
        print(uploaded_file.size)
    return render(request, 'upload.html')