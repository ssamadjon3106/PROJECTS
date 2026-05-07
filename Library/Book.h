#ifndef BOOK_H
#define BOOK_H

#include <iostream>
#include <string>
using namespace std;

template <typename T>
class Book {
private:
    T book_id;
    string title;
    string author;
    string genre;
    bool is_available;

public:
    // Constructors
    Book()
        : book_id(0), title(""), author(""), genre(""), is_available(true) {}

    Book(T id, const string& t, const string& a, const string& g, bool available = true)
        : book_id(id), title(t), author(a), genre(g), is_available(available) {}

    // Getters
    T getBookId() const { return book_id; }
    string getTitle() const { return title; }
    string getAuthor() const { return author; }
    string getGenre() const { return genre; }
    bool getAvailability() const { return is_available; }

    // Setters
    void setBookId(T id) { book_id = id; }
    void setTitle(const string& t) { title = t; }
    void setAuthor(const string& a) { author = a; }
    void setGenre(const string& g) { genre = g; }
    void setAvailability(bool available) { is_available = available; }

    // Overload operator==
    bool operator==(const Book& other) const {
        return this->book_id == other.book_id;
    }

    // Display book information
    void display_info() const {
        cout << "Book ID: " << book_id << ", Title: " << title
             << ", Author: " << author << ", Genre: " << genre
             << ", Available: " << (is_available ? "Yes" : "No") << endl;
    }

    // Mark book as borrowed
    void mark_as_borrowed() {
        is_available = false;
    }

    // Mark book as returned
    void mark_as_returned() {
        is_available = true;
    }
};

#endif // BOOK_H