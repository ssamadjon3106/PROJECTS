# LibraryHub

A comprehensive library management system built with Django, featuring book borrowing, seat reservations, and administrative dashboard capabilities.

## Features

### 📚 Book Management
- Browse available books with detailed information (title, author, ISBN, genre, description)
- Borrow and return books with due date tracking
- Automatic fine calculation for overdue books (500 per day)
- Track borrowing history and current loans

### 💺 Seat Reservations
- Interactive floor map with different study zones:
  - Quiet Zone (Q-101 to Q-105)
  - Group Zone (G-201 to G-204)
  - Computer Zone (C-301 to C-304)
  - Study Zone (S-401 to S-403)
  - Lounge Zone (L-501 to L-503)
- Reserve seats for specific time periods
- View and manage your reservations

### 👥 User Management
- Custom user roles: Students and Librarians
- User registration and authentication
- Profile management with activity tracking
- Points and streak system for engagement

### 🔍 Search Functionality
- Search books by title, author, or ISBN
- Real-time search results

### 📊 Librarian Dashboard
- Overview statistics (total books, students, active borrows, reservations)
- Overdue book tracking and management
- User management capabilities

## Technology Stack

- **Backend**: Django 5.2.12
- **Database**: SQLite3
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with responsive design

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd library_project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install django
   ```

4. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser (librarian account):**
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server:**
   ```bash
   python manage.py runserver
   ```

7. **Access the application:**
   - Open your browser and go to `http://127.0.0.1:8000`
   - Login with your superuser credentials or register as a student

## Usage

### For Students
1. **Register/Login** to create your account
2. **Browse Books** from the home page or search for specific titles
3. **Borrow Books** by clicking "Borrow" on available books
4. **Reserve Seats** using the interactive floor map
5. **View Your Books** and reservations in your profile

### For Librarians
1. **Access Dashboard** from the navigation menu
2. **Monitor Statistics** and overdue books
3. **Manage Users** and system settings through Django admin (`/admin/`)

## Project Structure

```
library_project/
├── accounts/          # User authentication and profiles
├── books/            # Book management and borrowing
├── dashboard/        # Librarian dashboard
├── search/           # Search functionality
├── seats/            # Seat reservation system
├── static/           # CSS and JavaScript files
├── templates/        # HTML templates
├── library_project/  # Django project settings
├── db.sqlite3        # SQLite database
└── manage.py         # Django management script
```

## Key Models

- **LibraryUser**: Custom user model with roles (Student/Librarian)
- **Book**: Book information with availability tracking
- **BorrowRecord**: Borrowing transactions with due dates
- **Seat**: Study seats with zones and positions
- **Reservation**: Seat booking records

## Pre-seeded Data

The system comes with pre-seeded data:
- **Books**: 10+ software engineering books
- **Seats**: 18 seats across 5 different zones

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

For questions or issues, please open an issue on the GitHub repository.
