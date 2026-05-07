#include <iostream>
#include <vector>
#include "Book.h"
#include "Member.h"
#include "Librarian.h"
#include "Library.h"
#include "Transaction.h"

using namespace std;

int main() {
    Library<int> library;
    Librarian<int> librarian(1, "Admin", "admin123");
    vector<Book<int>> books;
    vector<Transaction<int>> transactions;

    int choice;
    do {
        cout << "\n======= LIBRARY MENU =======\n";
        cout << "1. Add Book\n";
        cout << "2. View All Books\n";
        cout << "3. Register Member\n";
        cout << "4. View Registered Members\n";
        cout << "5. Issue Book\n";
        cout << "6. View Member's Borrowed Books\n";
        cout << "7. Search Book\n";
        cout << "8. View All Transactions\n";
        cout << "0. Exit\n";
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice) {
            case 1: {
                int id;
                string title, author, genre;
                cout << "Enter Book ID: ";
                cin >> id;
                cin.ignore();
                cout << "Enter Title: ";
                getline(cin, title);
                cout << "Enter Author: ";
                getline(cin, author);
                cout << "Enter Genre: ";
                getline(cin, genre);
                Book<int> book(id, title, author, genre);
                librarian.add_book(book, books);
                break;
            }
            case 2:
                librarian.view_all_books(books);
                break;

            case 3: {
                int id;
                string name, email;
                cout << "Enter Member ID: ";
                cin >> id;
                cin.ignore();
                cout << "Enter Name: ";
                getline(cin, name);
                cout << "Enter Email: ";
                getline(cin, email);
                Member<int> member(id, name, email);
                library.register_member(member);
                break;
            }

            case 4: {
                cout << "\nRegistered Members:\n";
                for (const auto& m : library.getMembers()) {
                    cout << m.getMemberId() << " - " << m.getName() << " (" << m.getEmail() << ")\n";
                }
                break;
            }

            case 5: {
                int bookId, memberId;
                cout << "Enter Book ID to issue: ";
                cin >> bookId;
                cout << "Enter Member ID: ";
                cin >> memberId;

                Book<int>* bookPtr = nullptr;
                Member<int>* memberPtr = nullptr;

                for (auto& b : books)
                    if (b.getBookId() == bookId) bookPtr = &b;

                for (auto& m : library.getMembers())
                    if (m.getMemberId() == memberId) memberPtr = &m;

                if (bookPtr && memberPtr) {
                    library.issue_book(*bookPtr, *memberPtr);

                    string today = "2025-04-14"; // You can prompt this if needed
                    string due = "2025-04-28";
                    Transaction<int> t(transactions.size() + 1, bookPtr->getTitle(), memberPtr->getName(), today, due);
                    transactions.push_back(t);
                    t.record_issue();
                } else {
                    cout << "Book or Member not found.\n";
                }
                break;
            }

            case 6: {
                int id;
                cout << "Enter Member ID: ";
                cin >> id;
                bool found = false;
                for (auto& m : library.getMembers()) {
                    if (m.getMemberId() == id) {
                        m.view_borrowed_books();
                        found = true;
                        break;
                    }
                }
                if (!found) cout << "Member not found.\n";
                break;
            }

            case 7: {
                string searchTitle;
                cout << "Enter title to search: ";
                cin.ignore();
                getline(cin, searchTitle);
                library.setBooks(books); // sync book list
                library.search_book(searchTitle);
                break;
            }

            case 8: {
                cout << "\nTransactions:\n";
                for (const auto& t : transactions)
                    t.record_issue();
                break;
            }

            case 0:
                cout << "Exiting... Goodbye!\n";
                break;

            default:
                cout << "Invalid choice. Try again.\n";
        }

    } while (choice != 0);

    return 0;
}
