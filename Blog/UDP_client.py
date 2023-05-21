import socket
import json
import sys

# Configure Parameters
SERVER = '192.168.0.101' # Enter the SERVER Address
PORT = 31415 # The server Port
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'


# ---API for the website---
def send(blogPiece: dict):
    if blogPiece != []:
        sendString = json.dumps(blogPiece)
        c.sendto(sendString.encode(FORMAT), ADDR)
    else:
        sendString = json.dumps('Fetch Data')
        c.sendto(sendString.encode(FORMAT), ADDR)

    allBlogPieces, addr = c.recvfrom(4096)
    if allBlogPieces != '':
        allBlogPieces = json.loads(allBlogPieces)
    else:
        allBlogPieces = []
    
    print(f'allBlogPieces: {allBlogPieces}')
    return allBlogPieces

def UDP_sendBlogPiece(blogPiece: dict):
    tmp = send(blogPiece)

def UDP_getBlogPieces():
    return send([])


# ---Build a socket---
c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)