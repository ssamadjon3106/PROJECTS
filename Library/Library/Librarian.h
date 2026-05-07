#ifndef LIBRARIAN_H
#define LIBRARIAN_H

#include "Book.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

template <typename T>
class Librarian  {
private:
    int librarian_id;
    string name;
    string password;

public:
    Librarian()
        : librarian_id(0), name(""), password("") {}

    Librarian(int id, const string& n, const string& pass)
        : librarian_id(id), name(n), password(pass) {}

    int getLibrarianId() const { return librarian_id; }
    string getName() const { return name; }
    string getPassword() const { return password; }

    void setLibrarianId(int id) { librarian_id = id; }
    void setName(const string& n) { name = n; }
    void setPassword(const string& pass) { password = pass; }

    void add_book(Book<T>& book, vector<Book<T>>& library_books) {
        library_books.push_back(book);
        cout << "Book '" << book.getTitle() << "' added to the library." << endl;
    }

    void remove_book(Book<T>& book, vector<Book<T>>& library_books) {
        library_books.erase(remove(library_books.begin(), library_books.end(), book), library_books.end());
        cout << "Book '" << book.getTitle() << "' removed from the library." << endl;
    }

    void view_all_books(vector<Book<T>>& library_books) {
        for (auto& book : library_books) {
            book.display_info();
        }
    }
};

#endif // LIBRARIAN_H
