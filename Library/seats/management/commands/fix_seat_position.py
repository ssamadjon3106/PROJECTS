import math
from django.core.management.base import BaseCommand
from seats.models import Seat


CX, CY = 500, 510

SEAT_POSITIONS = {
    # S zone: left wall diagonal
    "S-101": (40.0, 70.0),
    "S-102": (45.0, 70.0),
    "S-103": (40, 65.0),
    "S-104": (45, 65.0),
    "S-105": (55, 70.0),
    "S-106": (60, 70.0),
    "S-107": (55, 65),
    "S-108": (60, 65),

    # G zone: left inner arc
    "G-201": (87, 33.0),
    "G-202": (81, 20.0),
    "G-203": (64.0, 13.0),
    "G-204": (34, 13.0),

    # C zone: right inner arc
    "C-301": (72.0, 70.0),
    "C-302": (78.0, 62.0),
    "C-303": (83.0, 53.0),
    "C-304": (88.0, 44.0),
    
    # L zone: upper-right arc
    "L-501": (63.0, 22.0),
    "L-502": (70.0, 14.0),
    "L-503": (77.0,  7.0),
}


class Command(BaseCommand):
    help = "Rename Q seats to S seats, then reset pos_x/pos_y for all seats."

    def handle(self, *args, **options):
        # Step 1: rename Q-101..Q-105 to S-101..S-105
        renames = {
            "Q-101": "S-101",
            "Q-102": "S-102",
            "Q-103": "S-103",
            "Q-104": "S-104",
            "Q-105": "S-105",
            "S-401" : "S-106",
            "S-402" : "S-107",
            "S-403" : "S-108",
            "L-501" : "G-205",
            "L-502": "G-206"
        }
        for old, new in renames.items():
            count = Seat.objects.filter(seat_number=old).update(seat_number=new)
            if count:
                self.stdout.write(f"  Renamed {old} → {new}")

        # Step 2: update positions
        updated = 0
        missing = []
        for seat_number, (px, py) in SEAT_POSITIONS.items():
            try:
                seat = Seat.objects.get(seat_number=seat_number)
                seat.pos_x = px
                seat.pos_y = py
                seat.save(update_fields=["pos_x", "pos_y"])
                updated += 1
                self.stdout.write(f"  ✓ {seat_number} → ({px}%, {py}%)")
            except Seat.DoesNotExist:
                missing.append(seat_number)

        self.stdout.write(self.style.SUCCESS(f"\nUpdated {updated} seats."))
        if missing:
            self.stdout.write(self.style.WARNING(f"Not found: {', '.join(missing)}"))
