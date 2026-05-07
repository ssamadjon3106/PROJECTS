from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import hashlib


class UserRole(Enum):
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"
    PARENT = "parent"


class AbstractRole(ABC):

    def __init__(self, full_name, email, password, role):
        self._id = id(self)
        self._full_name = full_name
        self._email = email
        self.role = role
        self._password_hash = hashlib.sha256(password.encode()).hexdigest()
        self._created_at = datetime.now().isoformat()

    @abstractmethod
    def get_profile(self):
        pass

    @abstractmethod
    def update_profile(self, **kwargs):
        pass


class User(AbstractRole):

    def __init__(self, full_name, email, password, role):
        super().__init__(full_name, email, password, role)
        self._notifications = []

    def get_profile(self):
        return {
            "id": self._id,
            "full_name": self._full_name,
            "email": self._email,
            "role": self.role.value,
            "created_at": self._created_at
        }

    def update_profile(self, **kwargs):
        if "full_name" in kwargs:
            self._full_name = kwargs["full_name"]

        if "email" in kwargs:
            self._email = kwargs["email"]

        if "password" in kwargs:
            self._password_hash = hashlib.sha256(kwargs["password"].encode()).hexdigest()


class Student(User):

    def __init__(self, full_name, email, password, grade):
        super().__init__(full_name, email, password, UserRole.STUDENT)
        self.grade = grade
        self.assignments = {}

    def submit_assignment(self, assignment_id, content):

        if assignment_id in self.assignments:
            print("Already submitted")
            return False

        self.assignments[assignment_id] = {
            "content": content,
            "submitted_at": datetime.now().isoformat(),
            "grade": None,
            "comment": None
        }

        return True

    def view_grades(self):

        if not self.assignments:
            print("No assignments")
            return

        for aid, data in self.assignments.items():

            print("Assignment:", aid)
            print("Grade:", data["grade"])
            print("Comment:", data["comment"])

    def calculate_average_grade(self):

        grades = []

        for data in self.assignments.values():

            if isinstance(data["grade"], (int, float)):
                grades.append(data["grade"])

        if not grades:
            return 0

        return sum(grades) / len(grades)


class Teacher(User):

    def __init__(self, full_name, email, password, subjects):
        super().__init__(full_name, email, password, UserRole.TEACHER)
        self.subjects = subjects

    def grade_assignment(self, assignment_id, student_id, grade, comment, users):

        student = next((u for u in users if isinstance(u, Student) and u._id == student_id), None)

        if student and assignment_id in student.assignments:

            student.assignments[assignment_id]["grade"] = grade
            student.assignments[assignment_id]["comment"] = comment

            print("Graded successfully")
            return True

        print("Failed to grade")
        return False


class Parent(User):

    def __init__(self, full_name, email, password):
        super().__init__(full_name, email, password, UserRole.PARENT)
        self.children = {}

    def add_child(self, student):
        self.children[student._id] = student

    def view_child_grades(self, child_id):

        child = self.children.get(child_id)

        if not child:
            print("Child not found")
            return

        child.view_grades()


class Admin(User):

    def __init__(self, full_name, email, password):
        super().__init__(full_name, email, password, UserRole.ADMIN)
        self.users = {}

    def add_user(self, user, users):

        if user._id in self.users:
            return False

        self.users[user._id] = user
        users.append(user)

        return True

    def remove_user(self, user_id, users):

        user = self.users.pop(user_id, None)

        if user:
            users.remove(user)
            return True

        return False
