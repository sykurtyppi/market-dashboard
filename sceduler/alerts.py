"""
Alert system for sending notifications
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict
from datetime import datetime

class AlertManager:
    """Manages email/SMS alerts"""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = 587, 
                 email: str = None, password: str = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
    
    def send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email alert"""
        if not all([self.smtp_server, self.email, self.password]):
            print("⚠ Email not configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
            
            print(f"✓ Email sent to {to_email}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to send email: {e}")
            return False
    
    def send_signal_alert(self, signal_data: Dict, to_email: str):
        """Send alert for trading signal"""
        subject = f"🚨 {signal_data['signal_type']} Signal: {signal_data['signal']}"
        
        body = f"""
Market Risk Dashboard Alert
===========================

Signal Type: {signal_data['signal_type']}
Signal: {signal_data['signal']}
Strength: {signal_data.get('strength', 'N/A')}/100

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Details:
{signal_data.get('metadata', 'No additional details')}

---
This is an automated alert from your Market Risk Dashboard.
        """
        
        return self.send_email(to_email, subject, body)