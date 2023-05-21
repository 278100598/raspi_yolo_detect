import socket
import threading
import json


class BlogSever:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = (socket.gethostbyname(socket.gethostname()), 27418)
        self.users = {}
        # ---Storage---
        self.blogPieceId = 0 # Count the pieces of the blog
        self.blogDB = {} # Similar to Database
        self.db_lock = threading.Lock()
        # ---State---
        self.isOpen = True
        self.isOpen_lock = threading.Lock()

    def start_server(self):        
        try:
            self.socket.bind(self.addr)
        except Exception as e:
            print(e)

        self.socket.listen(5)
        print(f'[LISTENING] TCP Server is listening on {self.addr[0]}')
        print(f'[LISTENING] Port {self.addr[1]} can be connected')

        self.accept_cont()

    def accept_cont(self):
        while self.isOpen:
            try:
                c, addr = self.socket.accept()
            except Exception as e:
                break

            self.users[addr] = c
            number = len(self.users)
            print(f'[NEW CONNECTION] {addr} connected')
            print(f'[NUMBER OF CONNECTION] {number}')

            threading.Thread(target=self.recv_send, args=(c, addr)).start()
        
        print('[CLOSE THE SERVER]')
            
    def recv_send(self, conn, addr):
        while True:
            try:
                blogPiece = conn.recv(4096).decode('utf-8')
                if blogPiece != '':
                    blogPiece = json.loads(blogPiece)
                    print(f'Received: {blogPiece}')
                else:
                    print(f'[DISCONNECTION] {addr} closed')
                    conn.close()
                    self.users.pop(addr)

                    threading.Thread(target=self.confirm).start()
                    break

                # ---Lock the db variables---
                self.db_lock.acquire()

                if blogPiece == 'Fetch Data':
                    sendString = json.dumps(self.blogDB)
                    conn.send(sendString.encode('utf-8'))
                else:
                    self.blogDB[self.blogPieceId] = blogPiece
                    self.blogPieceId += 1

                    sendString = json.dumps(self.blogDB)
                    print(sendString)
                    conn.send(sendString.encode('utf-8'))
                
                self.db_lock.release()

            except ConnectionResetError:
                print(f'[DICONNECTION] {addr}')
                conn.close()
                self.users.pop(addr)
                break
    
    def confirm(self):
        self.isOpen_lock.acquire()

        if self.users == {} and self.isOpen:
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
                self.socket.close()

        self.isOpen_lock.release()


if __name__ == "__main__":
    sever = BlogSever()
    sever.start_server()