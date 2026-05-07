document.addEventListener('DOMContentLoaded', () => {
    const mapEl = document.getElementById('floor-map');
    if (!mapEl) return;

    const seats = JSON.parse(mapEl.dataset.seats);

    seats.forEach(seat => {
        const marker = document.createElement('div');
        marker.classList.add('seat-marker', seat.available ? 'available' : 'taken');
        marker.style.left = seat.pos_x + '%';
        marker.style.top = seat.pos_y + '%';
        marker.textContent = seat.number;
        marker.title = `${seat.zone} — ${seat.available ? 'Available' : 'Taken'}`;

        if (seat.available) {
            marker.addEventListener('click', () => {
                window.location.href = `/seats/${seat.id}/reserve/`;
            });
        }

        mapEl.appendChild(marker);
    });
});