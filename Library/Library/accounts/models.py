from django.db import models

# Create your models here.
from django.contrib.auth.models import AbstractUser
from django.db import models


class LibraryUser(AbstractUser):
    STUDENT = 'student'
    LIBRARIAN = 'librarian'
    ROLE_CHOICES = [
        (STUDENT, 'Student'),
        (LIBRARIAN, 'Librarian'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=STUDENT)
    points = models.PositiveIntegerField(default=0)
    streak_days = models.PositiveIntegerField(default=0)
    last_activity_date = models.DateField(null=True, blank=True)

    def is_librarian(self):
        return self.role == self.LIBRARIAN or self.is_superuser