import requests

msgPath = "http://www.pendola.net/api/port.php?action=publish&type=message&message="
dataPath = "http://www.pendola.net/api/port.php?action=publish&type=graph&message="
commandPath = "http://www.pendola.net/api/port.php?action="

def PublishMsg(msg):
    try:
        r = requests.get(msgPath + msg)
        if(r.status_code == requests.codes.ok):
            print("Publish successful")
        else:
            print("Publish failed")
    except:
        print("Publisher threw an exception")

def PublishData(value):
    try:
        r = requests.get(dataPath + str(value))
        if(r.status_code == requests.codes.ok):
            print("Publish successful")
        else:
            print("Publish failed")
    except:
        print("Publisher threw an exception")

def PublishCmd(cmd):
    try:
        r = requests.get(commandPath + str(cmd))
        if(r.status_code == requests.codes.ok):
            print("Publish successful")
        else:
            print("Publish failed")
    except:
        print("Publisher threw an exception")
