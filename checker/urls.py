from django.urls import path
from . import views

urlpatterns = [
    path('', views.check_plagiarism, name='check_plagiarism'),
    path("file-check/", views.file_check, name="file_check"),
    path("assignment-check/", views.assignment_check, name="assignment_check"),
    path('multi-file-check/', views.multi_file_check, name='multi_file_check'),
    path('research-paper-check/', views.research_paper_check, name='research_paper_check'),
]
