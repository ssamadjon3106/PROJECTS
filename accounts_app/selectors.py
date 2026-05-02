from django.contrib.auth import get_user_model


def get_active_students():
    return get_user_model().objects.filter(is_active=True).order_by('username')
