#ifndef LIBRARY_H
#define LIBRARY_H

#include <string>
#include <vector>
#include <iostream>
#include "Book.h"
#include "Member.h"
using namespace std;

template <typename T>
class Library {
private:
    vector<Book<T>> books;
    vector<Member<T>> members;

public:
    // Getters
    vector<Book<T>> getBooks() const { return books; }
    // Add this non-const version:
    vector<Member<T>>& getMembers() { return members; }

    // Keep your existing const version (for read-only use):
    const vector<Member<T>>& getMembers() const { return members; }


    // Setters
    void setBooks(const vector<Book<T>>& bks) { books = bks; }
    void setMembers(const vector<Member<T>>& mems) { members = mems; }

    // Search book by title
    void search_book(const string& search_title) const {
        for (const auto& book : books) {
            if (book.getTitle() == search_title) {
                book.display_info();
                return;
            }
        }
        cout << "Book not found." << endl;
    }

    // Register a member
    void register_member(const Member<T>& member) {
        members.push_back(member);
        cout << "Member '" << member.getName() << "' registered." << endl;
    }

    // Issue book to member
    void issue_book(Book<T>& book, Member<T>& member) {
        member.borrow_book(book);
    }
};

#endif // LIBRARY_H