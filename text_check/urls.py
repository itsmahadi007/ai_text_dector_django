from django.urls import path

from text_check.views import check_text

urlpatterns = [
    path(
        "check_text/",
        check_text,
        name="check_text",
    ),

]
