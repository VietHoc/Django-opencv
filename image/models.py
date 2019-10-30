from django.db import models

# Create your models here.
from django.db import models


class Image(models.Model):
    title = models.TextField()
    cover = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.title