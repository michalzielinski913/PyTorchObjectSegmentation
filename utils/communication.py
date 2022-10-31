import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from secret import SENDER_EMAIL, TO_EMAIL, SENDER_PASSWORD, EMAIL_HOST


def send_email(subject="Server Status", content="Calculations finished"):
    sender_email =SENDER_EMAIL
    receiver_email = TO_EMAIL
    password = SENDER_PASSWORD

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email

    # Create the plain-text and HTML version of your message
    text = content


    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(EMAIL_HOST, 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )
if __name__=="__main__":
    send_email()