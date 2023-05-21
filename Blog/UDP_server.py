import socket
import threading
import random
import json


class BlogSever:
    def __init__(self):
        # ---Storage---
        self.blogPieceId = 0 # Count the pieces of the blog
        self.blogDB = {} # Similar to Database
        self.db_lock = threading.Lock()
        # ---State---
        self.isOpen = True
        self.isOpen_lock = threading.Lock()
        # ---Count---
        self.maxCount = 6
        self.count = 0
        self.count_lock = threading.Lock()
            
    def start_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # addr = (socket.gethostbyname(socket.gethostname()), random.randint(1024, 65535))
        addr = (socket.gethostbyname(socket.gethostname()), 31415)
        s.bind(addr)
        # Wait for connection and Handle connection
        print(f'[LISTENING] UDP Server is listening on {addr[0]}')
        print(f'[LISTENING] Port {addr[1]} can be connected')
        
        while self.isOpen:
            if self.count < self.maxCount:
                try:
                    recvString, caddr = s.recvfrom(1024)
                    self.count_lock.acquire()
                    self.count += 1
                    self.count_lock.release()

                    print(f'[NEW CONNECTION] {caddr} connected')
                    print(f'[COUNT OF CONNECTION] {self.count}')
                    blogPiece = recvString.decode('utf-8')
                    if blogPiece != '':
                        blogPiece = json.loads(blogPiece)
                        print(f'Received: {blogPiece}')
                    else:
                        s.close()
                        break

                    # ---Lock the db variables---
                    self.db_lock.acquire()

                    if blogPiece == 'Fetch Data':
                        sendString = json.dumps(self.blogDB)
                        s.sendto(sendString.encode('utf-8'), caddr)
                    else:
                        self.blogDB[self.blogPieceId] = blogPiece
                        self.blogPieceId += 1

                        sendString = json.dumps(self.blogDB)
                        print(sendString)
                        s.sendto(sendString.encode('utf-8'), caddr)
                    
                    self.db_lock.release()

                except ConnectionResetError:
                    print(f'[DICONNECTION] {caddr}')
                    s.close()
                    self.users.pop(caddr)
                    break
            else:
                threading.Thread(target=self.confirm).start()

        print('[CLOSE THE SERVER]')
    
    def confirm(self):
        self.isOpen_lock.acquire()

        if self.count == self.maxCount and self.isOpen:
            fig = True # T: Continue, F: Quit
            while True:
                instruction = input('Do you want to continue?(y/n): ')
                if instruction == 'y' or instruction == 'Y':
                    fig = True
                    break
                elif instruction == 'n' or instruction == 'N':
                    fig = False
                    break
                else:
                    pass
            
            if fig:
                print('[CONTINUE]')
                pass
            else:
                self.isOpen = False
            
            self.count = 0

        self.isOpen_lock.release()


if __name__ == "__main__":
    sever = BlogSever()
    sever.start_server()