import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Mail:

    def __init__(self, Name, Roll_no, total_score, Result, Grade, email):
        self.Name = Name
        self.Roll_no = Roll_no
        self.total_score = total_score
        self.Result = Result
        self.Grade = Grade
        self.email = email


    def send_mail(self):

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()

        server.login('patelvini136@gmail.com','luspglgbqzncupym')

        subject = "Regards Performance of Student"

        body = f"Hi,\n\nBelow are details of student's performance based on previous test scores : \n\t> Name : {self.Name}\n\t> Roll no : {self.Roll_no}\n\t> Total score : {self.total_score}%\n\t> Grade : {self.Grade}\n\t> Result : {self.Result}\n\n\n\n Thank you."

        msg = f'Subject: {subject} \n\n{body}'

        server.sendmail('patelvini136@gmail.com', self.email,msg)

        server.quit()


