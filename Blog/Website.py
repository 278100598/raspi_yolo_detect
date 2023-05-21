from flask import Flask, render_template, request, redirect
import random
from TCP_client import TCP_sendBlogPiece, TCP_getBlogPieces
from UDP_client import UDP_sendBlogPiece, UDP_getBlogPieces

state = 0 # 0: TCP, 1: UDP

app = Flask('__name__')

@app.route('/', methods=['GET'])
def index():
    global state
    if state == 0:
        blog = TCP_getBlogPieces()
    elif state == 1:
        blog = UDP_getBlogPieces()
    
    if blog != []:
        return render_template('blog.html', blog=blog, blogIdList=list(map(str, range(len(blog)))))
    else:
        return render_template('blog.html')

@app.route('/submit', methods=['POST'])
def submit():
    blogPiece = {
        'username': request.form['username'],
        'title': request.form['title'],
        'message': request.form['message']
    }
    global state
    if state == 0:
        TCP_sendBlogPiece(blogPiece)
    elif state == 1:
        UDP_sendBlogPiece(blogPiece)
    
    return redirect('/')

@app.route('/tcp', methods=['GET'])
def tcp():
    global state
    state = 0
    return redirect('/')

@app.route('/udp', methods=['GET'])
def udp():
    global state
    state = 1
    return redirect('/')

if __name__ == '__main__':
    app.run(port=random.randint(1024, 65535))