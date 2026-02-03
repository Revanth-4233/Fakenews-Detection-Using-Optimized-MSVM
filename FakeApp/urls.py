from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="index"),
	             path("UserLogin.html", views.UserLogin, name="UserLogin"),
		     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		     path("Register.html", views.Register, name="Register"),
		     path("RegisterAction", views.RegisterAction, name="RegisterAction"),
		     path("LoadDataset", views.LoadDataset, name="LoadDataset"),
		     path("FeaturesSelection", views.FeaturesSelection, name="FeaturesSelection"),
		     path("RunML", views.RunML, name="RunML"),
		     path("Predict", views.Predict, name="Predict"),
		     path("PredictAction", views.PredictAction, name="PredictAction"),	
		     path("PredictFileAction", views.PredictFileAction, name="PredictFileAction"),	
		    ]