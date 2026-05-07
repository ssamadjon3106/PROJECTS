from django.db import migrations


def seed_seats(apps, schema_editor):
    Seat = apps.get_model('seats', 'Seat')

    seats = [
        # Quiet zone
        {'seat_number': 'Q-101', 'zone': 'quiet', 'pos_x': 10.0, 'pos_y': 70.0},
        {'seat_number': 'Q-102', 'zone': 'quiet', 'pos_x': 15.0, 'pos_y': 65.0},
        {'seat_number': 'Q-103', 'zone': 'quiet', 'pos_x': 20.0, 'pos_y': 60.0},
        {'seat_number': 'Q-104', 'zone': 'quiet', 'pos_x': 25.0, 'pos_y': 55.0},
        {'seat_number': 'Q-105', 'zone': 'quiet', 'pos_x': 30.0, 'pos_y': 50.0},

        # Group zone
        {'seat_number': 'G-201', 'zone': 'group', 'pos_x': 40.0, 'pos_y': 60.0},
        {'seat_number': 'G-202', 'zone': 'group', 'pos_x': 45.0, 'pos_y': 55.0},
        {'seat_number': 'G-203', 'zone': 'group', 'pos_x': 50.0, 'pos_y': 50.0},
        {'seat_number': 'G-204', 'zone': 'group', 'pos_x': 55.0, 'pos_y': 45.0},

        # Computer zone
        {'seat_number': 'C-301', 'zone': 'computer', 'pos_x': 70.0, 'pos_y': 55.0},
        {'seat_number': 'C-302', 'zone': 'computer', 'pos_x': 75.0, 'pos_y': 50.0},
        {'seat_number': 'C-303', 'zone': 'computer', 'pos_x': 80.0, 'pos_y': 45.0},
        {'seat_number': 'C-304', 'zone': 'computer', 'pos_x': 85.0, 'pos_y': 40.0},

        # Study zone (new)
        {'seat_number': 'S-401', 'zone': 'study', 'pos_x': 35.0, 'pos_y': 40.0},
        {'seat_number': 'S-402', 'zone': 'study', 'pos_x': 40.0, 'pos_y': 35.0},
        {'seat_number': 'S-403', 'zone': 'study', 'pos_x': 45.0, 'pos_y': 30.0},

        # Lounge zone (new)
        {'seat_number': 'L-501', 'zone': 'lounge', 'pos_x': 60.0, 'pos_y': 25.0},
        {'seat_number': 'L-502', 'zone': 'lounge', 'pos_x': 65.0, 'pos_y': 20.0},
        {'seat_number': 'L-503', 'zone': 'lounge', 'pos_x': 70.0, 'pos_y': 15.0},
    ]

    for s in seats:
        Seat.objects.get_or_create(
            seat_number=s['seat_number'],
            defaults={
                'zone': s['zone'],
                'pos_x': s['pos_x'],
                'pos_y': s['pos_y'],
                'is_active': True,
            },
        )


class Migration(migrations.Migration):
    dependencies = [
        ('seats', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(seed_seats),
    ]
