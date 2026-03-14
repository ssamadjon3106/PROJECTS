import json
from models import Student, Teacher, Parent, Admin


class StorageManager:

    FILE = "eduplatform_data.json"

    @staticmethod
    def save(users):

        data = []

        for user in users:

            user_data = {
                "id": user._id,
                "full_name": user._full_name,
                "email": user._email,
                "password_hash": user._password_hash,
                "role": user.role.value,
                "created_at": user._created_at
            }

            if isinstance(user, Student):

                user_data["grade"] = user.grade
                user_data["assignments"] = user.assignments

            if isinstance(user, Teacher):

                user_data["subjects"] = user.subjects

            if isinstance(user, Parent):

                user_data["children"] = list(user.children.keys())

            data.append(user_data)

        with open(StorageManager.FILE, "w") as f:

            json.dump(data, f, indent=4)

    @staticmethod
    def load():

        users = []

        try:

            with open(StorageManager.FILE, "r") as f:

                data = json.load(f)

            for u in data:

                role = u["role"]

                if role == "student":

                    user = Student(u["full_name"], u["email"], "temp", u["grade"])
                    user.assignments = u["assignments"]

                elif role == "teacher":

                    user = Teacher(u["full_name"], u["email"], "temp", u["subjects"])

                elif role == "parent":

                    user = Parent(u["full_name"], u["email"], "temp")

                elif role == "admin":

                    user = Admin(u["full_name"], u["email"], "temp")

                user._id = u["id"]
                user._password_hash = u["password_hash"]
                user._created_at = u["created_at"]

                users.append(user)

        except FileNotFoundError:

            pass

        return users
