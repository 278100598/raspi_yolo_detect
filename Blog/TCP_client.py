import socket
import threading
import json

# Configure Parameters
SERVER = '192.168.0.101' # Enter the SERVER Address
PORT = 27418 # The server Port
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'

# ---API for the website---
def send(blogPiece: dict):
    if blogPiece != []:
        sendString = json.dumps(blogPiece)
        c.send(sendString.encode(FORMAT))
    else:
        sendString = json.dumps('Fetch Data')
        c.send(sendString.encode(FORMAT))

    allBlogPieces = c.recv(4096).decode(FORMAT)
    if allBlogPieces != '':
        allBlogPieces = json.loads(allBlogPieces)
    else:
        allBlogPieces = []
    
    print(f'allBlogPieces: {allBlogPieces}')
    return allBlogPieces

def TCP_sendBlogPiece(blogPiece: dict):
    tmp = send(blogPiece)

def TCP_getBlogPieces():
    return send([])
# ---------------

# ---Build a socket---
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def tryToConnect():
    # ---Connect to the server---
    isConnect = False
    print('[Start Connecting to Server]')
    while not isConnect:
        try:
            c.connect(ADDR)
            print('[Client Connects to Server]')
            isConnect = True
        except:
            continue

threading.Thread(target=tryToConnect).start()