#ifndef MEMBER_H
#define MEMBER_H

#include <iostream>
#include <vector>
#include <string>
#include "Book.h"
using namespace std;

template <typename T>
class Member  {
private:
    int member_id;
    string name;
    string email;
    vector<Book<T>> borrowed_books;

public:
    // Constructors
    Member()
        : member_id(0), name(""), email("") {}

    Member(int id, const string& n, const string& e)
        : member_id(id), name(n), email(e) {}

    // Getters
    int getMemberId() const { return member_id; }
    string getName() const { return name; }
    string getEmail() const { return email; }
    vector<Book<T>> getBorrowedBooks() const { return borrowed_books; }

    // Setters
    void setMemberId(int id) { member_id = id; }
    void setName(const string& n) { name = n; }
    void setEmail(const string& e) { email = e; }
    void setBorrowedBooks(const vector<Book<T>>& books) { borrowed_books = books; }

    // Borrow book
    void borrow_book(Book<T>& book) {
        if (book.getAvailability()) {
            book.mark_as_borrowed();
            borrowed_books.push_back(book);
            cout << "Book '" << book.getTitle() << "' borrowed by " << name << endl;
        } else {
            cout << "Book '" << book.getTitle() << "' is not available." << endl;
        }
    }

    // View borrowed books
    void view_borrowed_books() const {
        if (borrowed_books.empty()) {
            cout << name << " has not borrowed any books." << endl;
        } else {
            for (const auto& book : borrowed_books) {
                cout << "Borrowed Book: " << book.getTitle() << endl;
            }
        }
    }
};

#endif // MEMBER_H