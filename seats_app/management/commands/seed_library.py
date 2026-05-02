from django.core.management.base import BaseCommand

from library.models import Book, Seat


class Command(BaseCommand):
    help = 'Create sample books for the library demo.'

    def handle(self, *args, **options):
        books = [
            {
                'title': 'Clean Code',
                'author': 'Robert C. Martin',
                'isbn': '9780132350884',
                'category': 'Software Engineering',
                'shelf_location': 'A1',
                'total_copies': 3,
                'available_copies': 3,
            },
            {
                'title': 'Introduction to Algorithms',
                'author': 'Thomas H. Cormen',
                'isbn': '9780262033848',
                'category': 'Algorithms',
                'shelf_location': 'B2',
                'total_copies': 2,
                'available_copies': 2,
            },
            {
                'title': 'Django for Beginners',
                'author': 'William S. Vincent',
                'isbn': '9781735467207',
                'category': 'Web Development',
                'shelf_location': 'C3',
                'total_copies': 4,
                'available_copies': 4,
            },
            {
                'title': 'Database System Concepts',
                'author': 'Abraham Silberschatz',
                'isbn': '9780073523323',
                'category': 'Databases',
                'shelf_location': 'D1',
                'total_copies': 2,
                'available_copies': 2,
            },
        ]

        created = 0
        for data in books:
            _, was_created = Book.objects.get_or_create(isbn=data['isbn'], defaults=data)
            created += int(was_created)

        seats = [
            ('A1', 'Entrance Left', 1, 4),
            ('A2', 'Entrance Left', 1, 5),
            ('A3', 'Entrance Left', 1, 6),
            ('A4', 'Entrance Left', 2, 4),
            ('A5', 'Entrance Left', 2, 5),
            ('A6', 'Entrance Left', 2, 6),
            ('B1', 'Entrance Center', 3, 4),
            ('B2', 'Entrance Center', 3, 5),
            ('B3', 'Entrance Center', 3, 6),
            ('B4', 'Entrance Center', 4, 4),
            ('B5', 'Entrance Center', 4, 5),
            ('B6', 'Entrance Center', 4, 6),
            ('C1', 'Entrance Right', 5, 4),
            ('C2', 'Entrance Right', 5, 5),
            ('C3', 'Entrance Right', 5, 6),
            ('C4', 'Entrance Right', 6, 4),
            ('C5', 'Entrance Right', 6, 5),
            ('C6', 'Entrance Right', 6, 6),
        ]

        seat_created = 0
        for label, zone, row, column in seats:
            _, was_created = Seat.objects.get_or_create(
                label=label,
                defaults={
                    'zone': zone,
                    'row': row,
                    'column': column,
                },
            )
            seat_created += int(was_created)

        self.stdout.write(self.style.SUCCESS(f'Seed complete. Created {created} books and {seat_created} seats.'))
