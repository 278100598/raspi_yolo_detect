import socket

HOST = '10.115.52.199'
PORT = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(5)

print('server start at: %s:%s' % (HOST, PORT))
print('wait for connection...')


con, addr = s.accept()
print('connected by ' + str(addr))
file_name = con.recv(1024).decode()

with open('new'+file_name,'wb') as f:
    while True:
        indata = con.recv(1024)
        f.write(indata)
        if len(indata) == 0: # connection closed
            con.close()
            print('client closed connection.')
            break


s.close()