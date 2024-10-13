# MIT License
#
# Copyright (c) 2024 Driveline Research
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Kyle Boddy, Driveline Research

import requests
import json
import logging
import sys

def send_notification(channel, message, webhook_url):
    """
    Sends a notification to a specified channel using a webhook URL.
    
    Args:
    channel (str): The channel to send the notification to.
    message (str): The message content of the notification.
    webhook_url (str): The URL of the webhook to use for sending the notification.
    
    Returns:
    requests.Response: The response object from the HTTP request.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    payload = {
        'channel': channel,
        'text': message
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        logger.info("Notification sent successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send notification. Error: {e}")
        return None
    
    return response

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python send_notification.py <channel> <message> <webhook_url>")
        sys.exit(1)
    
    channel = sys.argv[1]
    message = sys.argv[2]
    webhook_url = sys.argv[3]
    
    send_notification(channel, message, webhook_url)
