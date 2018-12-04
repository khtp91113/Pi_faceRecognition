#coding:utf-8
import socket
import os
import sys
import picamera
import struct
import io
import time
import dns.resolver
import threading
DNS = 'pi.service'
SERVER_IP = ''
PORT = 20000
BUFFER_SIZE = 1024
interval = 5
last = {}

def create_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            print('trying connect to server')
            sock.connect((SERVER_IP, PORT))
            break
        except:
            time.sleep(10)
    print('socket connected')
    return sock

def send_image(sock):
    connection = sock.makefile('wb')
    try:
        with picamera.PiCamera() as camera:
            camera.resolution = (640, 480)
            camera.start_preview()
            time.sleep(2)

            stream = io.BytesIO()
            while True:
                for foo in camera.capture_continuous(stream, 'jpeg'):
                    connection.write(struct.pack('<L', stream.tell()))
                    connection.flush()
                    stream.seek(0)
                    connection.write(stream.read())
                    stream.seek(0)
                    stream.truncate()
                    result = sock.recv(BUFFER_SIZE)
                    print(result)
                    result = result.decode('utf-8')
                    if result != 'No face':
                        speak(result)

        connection.write(struct.pack('<L', 0))
    finally:
        connection.close()
        sock.close()

def speak(result):
    names = result.split(',')
    count = 0
    lis = []
    for name in names:
        if name == 'unknown':
            continue
        if name not in last or time.time() - last[name] > interval:
            last[name] = time.time()
            count += 1
            lis.append(name)
    
    if count == 0:
        return
    string = "hi"
    for name in lis:
        string += ", " + name
    thread = myThread(string)
    thread.start()

class myThread(threading.Thread):
    def __init__(self, string):
        threading.Thread.__init__(self)
        self.string = string
    def run(self):
        os.system('espeak "' + self.string + '" -vzh+f3 -a 200')

def dns_query():
    resolver = dns.resolver.Resolver()
    try:
        answer = resolver.query(DNS)
    except:
        return False
    global SERVER_IP
    SERVER_IP = str(answer[0])
    return True

def main():
    print('dns ' + DNS)
    while True:
        print('trying get dns')
        if dns_query() == True:
            break
        time.sleep(5) 
    print('server ip ' + SERVER_IP)
    sock = create_socket()
    send_image(sock)

if __name__ == '__main__':
    main()
