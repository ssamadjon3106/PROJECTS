from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Legacy seed command (moved to seats_app)'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('This command has been moved to: python manage.py seed_library (via seats_app)'))
