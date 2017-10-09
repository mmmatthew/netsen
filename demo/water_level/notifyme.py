# This script requires Python 3

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def get_settings(filename):
    """
    Return settings dict
    """

    settings = {}
    with open(filename, mode='r') as file:
        for s in file:
            s = s.strip('\n\r')
            settings[s.split('=')[0]]=s.split('=')[1]
    return settings

def notifyme(settingsfile, message, subject='Automatic notification'):
    """
    settingsfile contains: host, port, user, password
    """

    settings = get_settings(settingsfile)

    # set up the SMTP server
    s = smtplib.SMTP(host=settings['host'], port=int(settings['port']))
    s.starttls()
    s.login(settings['user'], settings['password'])


    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template
    message = message

    # Prints out the message body for our sake
    print(message)

    # setup the parameters of the message
    msg['From']=settings['user']
    msg['To']=settings['to_addr']
    msg['Subject']=subject
    
    # add in the message body
    msg.attach(MIMEText(message, 'plain'))
    
    # send the message via the server set up earlier.
    s.send_message(msg)
    del msg
        
    # Terminate the SMTP session and close the connection
    s.quit()
    
