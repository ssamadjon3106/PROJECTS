from models import *
from storage_manager import StorageManager
from data_exporter import DataExporter
import hashlib


def main():

    print("EduPlatform CLI")

    users = StorageManager.load()

    if not any(isinstance(u, Admin) for u in users):

        admin = Admin("Samadjon Sayfullayev", "admin@mail.com", "123")
        users.append(admin)

    while True:

        print("\n1 Admin")
        print("2 Teacher")
        print("3 Student")
        print("4 Parent")
        print("5 Exit")

        role = input("Role: ")

        if role == "1":

            uid = int(input("Admin ID: "))
            pwd = input("Password: ")

            admin = next((u for u in users if isinstance(u, Admin) and u._id == uid), None)

            if not admin or admin._password_hash != hashlib.sha256(pwd.encode()).hexdigest():

                print("Invalid login")
                continue

            print("1 Add user")
            print("2 Remove user")

            action = input("Action: ")

            if action == "1":

                name = input("Name: ")
                email = input("Email: ")
                password = input("Password: ")

                print("1 Student")
                print("2 Teacher")
                print("3 Parent")

                t = input("Type: ")

                if t == "1":

                    grade = input("Grade: ")
                    user = Student(name, email, password, grade)

                elif t == "2":

                    subjects = input("Subjects: ").split(",")
                    user = Teacher(name, email, password, subjects)

                elif t == "3":

                    user = Parent(name, email, password)

                else:

                    print("Invalid")
                    continue

                admin.add_user(user, users)

                StorageManager.save(users)

                print("User added")

            elif action == "2":

                uid = int(input("User ID: "))

                admin.remove_user(uid, users)

                StorageManager.save(users)

        elif role == "5":

            StorageManager.save(users)

            DataExporter.export_to_csv(users)

            print("Goodbye")

            break


if __name__ == "__main__":

    main()
