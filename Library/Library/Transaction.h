#ifndef TRANSACTION_H
#define TRANSACTION_H

#include <iostream>
#include <string>
using namespace std;

template <typename T>
class Transaction {
private:
    T transaction_id;
    string book_title;
    string member_name;
    string date_issued;
    string date_due;
    string date_returned;

public:
    Transaction()
        : transaction_id(0), book_title(""), member_name(""),
          date_issued(""), date_due(""), date_returned("") {}

    Transaction(T id, const string& book, const string& member,
                const string& issued, const string& due, const string& returned = "")
        : transaction_id(id), book_title(book), member_name(member),
          date_issued(issued), date_due(due), date_returned(returned) {}

    // Getters
    T getTransactionId() const { return transaction_id; }
    string getBookTitle() const { return book_title; }
    string getMemberName() const { return member_name; }
    string getDateIssued() const { return date_issued; }
    string getDateDue() const { return date_due; }
    string getDateReturned() const { return date_returned; }

    // Setters
    void setTransactionId(T id) { transaction_id = id; }
    void setBookTitle(const string& title) { book_title = title; }
    void setMemberName(const string& name) { member_name = name; }
    void setDateIssued(const string& date) { date_issued = date; }
    void setDateDue(const string& date) { date_due = date; }
    void setDateReturned(const string& date) { date_returned = date; }

    // Other member functions
    void record_issue() const {
        cout << "Transaction ID: " << transaction_id << endl;
        cout << "Book: '" << book_title << "' issued to " << member_name << endl;
        cout << "Date Issued: " << date_issued << endl;
        cout << "Due Date: " << date_due << endl;
    }

    void record_return() const {
        cout << "Transaction ID: " << transaction_id << endl;
        cout << "Book: '" << book_title << "' returned by " << member_name << endl;
        cout << "Date Returned: " << date_returned << endl;
    }
};

#endif // TRANSACTION_H