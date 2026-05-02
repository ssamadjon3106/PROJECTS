const isbnInput = document.querySelector('input[name="isbn"]');

if (isbnInput) {
    isbnInput.addEventListener('input', () => {
        isbnInput.value = isbnInput.value.replace(/\s/g, '');
    });
}

const themeToggle = document.querySelector('[data-theme-toggle]');
const savedTheme = localStorage.getItem('library-theme');

if (savedTheme === 'dark') {
    document.body.classList.add('dark');
} else if (savedTheme === 'light') {
    document.body.classList.remove('dark');
}

if (themeToggle) {
    const syncThemeLabel = () => {
        themeToggle.textContent = document.body.classList.contains('dark') ? 'Light' : 'Dark';
    };

    syncThemeLabel();
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark');
        localStorage.setItem('library-theme', document.body.classList.contains('dark') ? 'dark' : 'light');
        syncThemeLabel();
    });
}

const seats = document.querySelectorAll('.seat.available, .entrance-seat.available');
const selectedSeatLabel = document.querySelector('#selected-seat');
const selectedSeatId = document.querySelector('[data-selected-seat-id]');
const selectedZone = document.querySelector('#selected-zone');
const selectedStatus = document.querySelector('#selected-status');
const selectedOwner = document.querySelector('#selected-owner');
const reserveButton = document.querySelector('[data-reserve-button]');

seats.forEach((seat) => {
    seat.addEventListener('click', () => {
        document.querySelectorAll('.seat, .entrance-seat').forEach((item) => item.classList.remove('selected'));
        seat.classList.add('selected');

        if (selectedSeatLabel) {
            selectedSeatLabel.textContent = seat.dataset.seat;
        }

        if (selectedSeatId) {
            selectedSeatId.value = seat.dataset.seatId || '';
        }

        if (selectedZone) {
            selectedZone.textContent = seat.dataset.zone || '--';
        }

        if (selectedStatus) {
            selectedStatus.textContent = seat.dataset.status || 'available';
        }

        if (selectedOwner) {
            selectedOwner.textContent = seat.dataset.reservedBy || '--';
        }

        if (reserveButton) {
            reserveButton.disabled = false;
            reserveButton.textContent = `Reserve ${seat.dataset.seat}`;
        }
    });
});

const clock = document.querySelector('[data-clock]');

if (clock) {
    const updateClock = () => {
        clock.textContent = new Intl.DateTimeFormat([], {
            hour: '2-digit',
            minute: '2-digit',
        }).format(new Date());
    };

    updateClock();
    setInterval(updateClock, 1000 * 30);
}

const liveNowButton = document.querySelector('[data-live-now]');

if (liveNowButton) {
    liveNowButton.addEventListener('click', () => {
        const dateInput = document.querySelector('input[name="reservation_date"]');
        const timeInput = document.querySelector('input[name="start_time"]');
        const now = new Date();

        if (dateInput) {
            dateInput.value = now.toISOString().slice(0, 10);
        }

        if (timeInput) {
            timeInput.value = now.toTimeString().slice(0, 5);
        }
    });
}
