import requests

msgPath = "http://www.pendola.net/api/port.php?action=publish&type=message&message="
dataPath = "http://www.pendola.net/api/port.php?action=publish&type=graph&message="

def PublishMsg(msg):
    r = requests.get(msgPath + msg)
    if(r.status_code == requests.codes.ok):
        print("Publish successful")
    else:
        print("Publish failed")

def PublishData(value):
    r = requests.get(dataPath + str(value))
    if(r.status_code == requests.codes.ok):
        print("Publish successful")
    else:
        print("Publish failed")
