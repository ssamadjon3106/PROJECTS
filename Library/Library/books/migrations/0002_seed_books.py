from django.db import migrations


def seed_books(apps, schema_editor):
    Book = apps.get_model('books', 'Book')

    books = [
        {
            'title': 'The Pragmatic Programmer',
            'author': 'Andrew Hunt & David Thomas',
            'isbn': '9780135957059',
            'genre': 'Software Engineering',
            'description': 'A practical guide to becoming a better programmer.',
            'total_copies': 3,
            'available_copies': 3,
        },
        {
            'title': 'Clean Code',
            'author': 'Robert C. Martin',
            'isbn': '9780132350884',
            'genre': 'Software Craftsmanship',
            'description': 'A handbook of agile software craftsmanship and principles of writing clean code.',
            'total_copies': 4,
            'available_copies': 4,
        },
        {
            'title': 'The Clean Coder',
            'author': 'Robert C. Martin',
            'isbn': '9780137081073',
            'genre': 'Software Engineering',
            'description': 'A code of conduct for professional developers—how to practice clean development.',
            'total_copies': 2,
            'available_copies': 2,
        },
        {
            'title': 'Refactoring',
            'author': 'Martin Fowler',
            'isbn': '9780201619576',
            'genre': 'Software Engineering',
            'description': 'Improving the design of existing code without changing its external behavior.',
            'total_copies': 3,
            'available_copies': 3,
        },
        {
            'title': 'Design Patterns',
            'author': 'Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides',
            'isbn': '9780201633610',
            'genre': 'Software Design',
            'description': 'Elements of reusable object-oriented software design.',
            'total_copies': 3,
            'available_copies': 3,
        },
        {
            'title': 'Introduction to Algorithms',
            'author': 'Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein',
            'isbn': '9780262033848',
            'genre': 'Algorithms',
            'description': 'A comprehensive textbook covering a broad range of algorithms.',
            'total_copies': 2,
            'available_copies': 2,
        },
        {
            'title': 'Effective Java',
            'author': 'Joshua Bloch',
            'isbn': '9780134685991',
            'genre': 'Programming',
            'description': 'Best practices for writing robust, maintainable Java.',
            'total_copies': 3,
            'available_copies': 3,
        },
        {
            'title': 'Head First Design Patterns',
            'author': 'Eric Freeman & Elisabeth Robson',
            'isbn': '9780596007126',
            'genre': 'Software Design',
            'description': 'Learn design patterns in an engaging, visual way.',
            'total_copies': 2,
            'available_copies': 2,
        },
        {
            'title': 'Code Complete',
            'author': 'Steve McConnell',
            'isbn': '9780735619678',
            'genre': 'Software Engineering',
            'description': 'A practical handbook for building better software.',
            'total_copies': 2,
            'available_copies': 2,
        },
        {
            'title': 'Domain-Driven Design',
            'author': 'Eric Evans',
            'isbn': '9780321125217',
            'genre': 'Software Architecture',
            'description': 'Tackling complexity in the heart of software with domain-driven design.',
            'total_copies': 2,
            'available_copies': 2,
        },
    ]

    for b in books:
        Book.objects.get_or_create(
            isbn=b['isbn'],
            defaults={
                'title': b['title'],
                'author': b['author'],
                'genre': b.get('genre', ''),
                'description': b.get('description', ''),
                'total_copies': b.get('total_copies', 1),
                'available_copies': b.get('available_copies', b.get('total_copies', 1)),
            },
        )


class Migration(migrations.Migration):
    dependencies = [
        ('books', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(seed_books),
    ]
