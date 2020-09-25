# inspired by https://realpython.com/python-send-email/
# for better security: setup ssl connection to gmail as in link above
# 
# run the following command to keep the script running in backgroundeven if you exit the shell (nohup + &)
# - std.out and std.err are redirected to error_out.log in case your script crashes
#
# > nohup python script_template.py 2> error_out.log &
#

import smtplib
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
file_handler = logging.FileHandler(filename='experiment.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

src_email = XXX
src_email_pass = YYY
dst_email = ZZZ

def notification_email(content_str):
    mail = smtplib.SMTP('smtp.gmail.com',587)
    content = 'Subject: %s\n\n%s' % ('script', content_str)
    mail.ehlo()
    mail.starttls()
    mail.login(src_email, src_email_pass)
    mail.sendmail(src_email, dst_email, content) 
    mail.close()
    logger.info(content_str)
    
try:
    logger.info('start')
    print('hello world')
    # 
    # COMPUTATION
    #
    
    notification_email('finished')

except:
    notification_email('crashed')
