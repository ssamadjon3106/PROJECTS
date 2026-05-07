import csv
from datetime import datetime
from openpyxl import Workbook
from models import Student, Teacher, Parent


class DataExporter:

    @staticmethod
    def export_to_csv(users, filename="export"):

        with open(f"{filename}_users.csv", "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["ID", "Name", "Email", "Role"])

            for user in users:

                writer.writerow([
                    user._id,
                    user._full_name,
                    user._email,
                    user.role.value
                ])

        print("CSV export completed")

    @staticmethod
    def export_to_xlsx(users, filename="export.xlsx"):

        wb = Workbook()

        ws = wb.active
        ws.title = "Users"

        ws.append(["ID", "Name", "Email", "Role"])

        for user in users:

            ws.append([
                user._id,
                user._full_name,
                user._email,
                user.role.value
            ])

        wb.save(filename)

        print("XLSX export completed")
