# Run this in python manage.py shell
from seats.models import Seat

renames = {
    "Q-101": "S-101",
    "Q-102": "S-102",
    "Q-103": "S-103",
    "Q-104": "S-104",
    "Q-105": "S-105",
}
for old, new in renames.items():
    Seat.objects.filter(seat_number=old).update(seat_number=new)