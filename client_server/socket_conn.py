# coding=utf8
# author: luruiyuan
# v0.1 date: 2018-09-18
# file: socket_conn.py

import sys
import os
import platform
import json
import selectors
import socket
import threading
import multiprocessing
from multiprocessing.queues import Empty as QEmptyException, Full as QFullException

def _find_N_free_ports(ip=None, start_port=7000, N=1, used_ports_set=None):
    '''
    Return the list of N continuous ports from the start port.

    If `ip` is None, the port that any ip address that use the given port will be seen as `In use`.
    Otherwise, only specified `ip` address that use port will be seen as `In use`

    Note that users should call len() to obtain actual number of ports, in case
    of no enough ports found.
    '''
    err = 'Illigal start port: \'%d\' and N: \'$d\'\n(start_port > 1024) and (N > 0) and (start_port + N < 65536) is required.'
    assert start_port > 1024 and N > 0 and start_port + N < 65536, err
    sys2cmds = {
        'windows': 'an | findstr :%s',
        'linux': 'na | grep :%s',
        'sunos': 'an | grep "\. %s"',
        'hp':'an | grep "\. %s"',
        'aix': 'Aan | grep "\. %s"',
    }
    system = platform.system().lower()
    assert system in sys2cmds, 'Unsupported system: \'%s\'' % system
    available = []
    for port in range(start_port, 65536):
        if used_ports_set is None or port not in used_ports_set:
            ip_port = '%s' % port if ip is None else '*%s:%s' % (ip, port) # if ip addr is specified
            cmd = 'netstat -%s' % (sys2cmds[system] % ip_port)
            if not os.popen(cmd).readlines(): # get [] if port is available
                available.append(port)
                if len(available) >= N:
                    break
    return available

def _create_socket(islocal=False, IPv=4, isTCP=True, sock_blocking=False, timeout=0):
    '''Create socket by settings'''
    family = _get_family(IPv, islocal)
    socktype = socket.SOCK_STREAM if isTCP else socket.SOCK_DGRAM
    sock = socket.socket(family=family, type=socktype)
    sock.settimeout(timeout)
    sock.setblocking(sock_blocking)
    return sock

def _get_family(IPversion, islocal):
    '''Return Protocol family according to given IPversion and if local flag'''
    if int(IPversion) == 4:
        family = socket.AF_UNIX if islocal and hasattr(socket, 'AF_UNIX') else socket.AF_INET
    elif int(IPversion) == 6:
        family = socket.AF_INET6
    else:
        raise NotImplementedError("Not supported IP version: 'IPv%s'.\nOnly IPv==(4 or 6) are supported" % repr(IPversion))
    return family

def _get_selector():
    '''Return selector according to system.'''
    return selectors.DefaultSelector() if os.name == 'nt' else selectors.DefaultSelector()

def _get_current_host_ip():
    '''Return the ip address of current host'''
    sock = _create_socket(isTCP=False) # UDP socket
    try:
        sock.connect(('8.8.8.8', 80))
        ip = sock.getsockname()[0] # get (ip, port), only ip needed
    finally:
        sock.close()
    return ip


class SOCKET_PROCESSOR(object):
    '''
    Processing socket related info
    '''

    def __init__(self, islocal=False, IPv=4, isTCP=True, sock_blocking=False,
                 timeout=0, buffsize=2048, reuse_port=True):
        '''Initialize socket. Default is to create a nonblocking socket'''
        self.isopen = True
        self.buffsize = buffsize if buffsize > 0 else 2048
        self.sock = SOCKET_PROCESSOR._get_socket(islocal, IPv, isTCP, sock_blocking, timeout, reuse_port)
        self.reuse_port = reuse_port
        self.islocal = islocal
        self.IPv = IPv
        self.isTCP = isTCP
        self.sock_blocking = sock_blocking
        self.timeout = timeout
        self.buffsize = buffsize
        self.cur_conn_addr = None
        self.remained_bytes = None

    def __del__(self):
        '''Deconstruct Function'''
        self.close()
    
    @classmethod
    def _get_socket(cls, islocal, IPv, isTCP, sock_blocking, timeout, reuse_port):
        sock = _create_socket(islocal, IPv, isTCP, sock_blocking, timeout)
        if reuse_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock
    
    def connect_to_new_addr(self, new_addr):
        if self.cur_conn_addr != new_addr:
            del self.sock
            self.sock = self._get_socket(self.islocal, self.IPv, self.isTCP,
                                        self.sock_blocking, self.timeout, self.reuse_port)
            self.connect(new_addr)

    def bind(self, bind_addr):
        '''
        bind socket to given ip and port
        '''
        self.sock.bind(bind_addr)

    def listen(self, backlog=10):
        '''start listen'''
        self.sock.listen(backlog)
    
    def connect(self, addr):
        '''connect to given ip address and port'''
        self.sock.connect(addr)
    
    def accept(self):
        '''Accept a request from client'''
        new_sock, addr = self.sock.accept()
        return new_sock, addr
    
    def send(self, data, type='data', encoding='utf8'):
        '''
        Send all data bytes through given connection
        Send method will add 4 bytes int in the head of the package indicating the length
        of packets.
        '''
        if (encoding is not None) and isinstance(data, str):
            data = data.encode(encoding)
        else:
            raise ValueError("Data type is '%s' and encoding is '%s'\nData: bytes<==>encoding: None | Data: str<==>encoding: not None" % (type(data), encoding))
        # descript data bytes for remote host
        json_dscrpt = json.dumps({'type': type, 'encoding': encoding}, ensure_ascii=False).encode('utf8')
        # send description and data
        # 长度标签, 用无符号4字节整数表示在数据之前
        des_len = len(json_dscrpt).to_bytes(length=4, byteorder=sys.byteorder, signed=False)
        data_len = len(data).to_bytes(length=4, byteorder=sys.byteorder, signed=False)
        self.sock.sendall(b''.join([des_len, json_dscrpt, data_len, data]))

    def recv(self, sock, mask):
        '''
        Receive data from other client
        '''
        # processing exit case
        data, remained = self._recv_until_length(sock, length=1, remained_bytes=self.remained_bytes)
        if data is None:
            return
        data_1, remained = self._recv_until_length(sock, length=3, remained_bytes=remained)
        # get json length
        json_len = int.from_bytes(data+data_1, sys.byteorder, signed=False)
        # get real json bytes and transform to json
        des_bytes, remained = self._recv_until_length(sock, length=json_len, remained_bytes=remained)
        description = json.loads(des_bytes.decode('utf8'))
        data_type = description['type']
        if data_type == 'signal':
            self.remained_bytes = remained
            return (data_type, description['signal'])
        elif data_type == 'data':
            encoding = description['encoding']
            # more data should be received
            data_len, remained = self._recv_until_length(sock, length=4, remained_bytes=remained)
            data_len = int.from_bytes(data_len, sys.byteorder, signed=False)
            recved, remained = self._recv_until_length(sock, length=data_len, remained_bytes=remained)
            # set remained attr for next invoke
            self.remained_bytes = remained
            return recved if encoding is None else recved.decode(encoding)
        else:
            self.remained_bytes = remained
            raise NotImplementedError("Received data type '%r' not supported." % data_type)

    def _recv_until_length(self, sock, length, remained_bytes=None):
            total_len = 0
            total_recved = []
            # processing remained bytes
            if remained_bytes is not None:
                remain_len = len(remained_bytes)
                if remain_len >= length:
                    return remained_bytes[:length], (None if remain_len == length else remained_bytes[length:])
                else:
                    total_recved.append(remained_bytes)
                    total_len += remain_len
            # if the len of remained bytes is shorter than length, receive new bytes
            while total_len < length:
                try:
                    reply = sock.recv(self.buffsize)
                except BlockingIOError:
                    continue
                # process close connection case
                if reply == b'':
                    return None, None
                total_len += len(reply)
                total_recved.append(reply)
            # calculate remained bytes
            remain_len = total_len - length
            last_idx = len(total_recved[-1]) - remain_len
            remained = None if remain_len == 0 else total_recved[-1][last_idx:]
            total_recved[-1] = total_recved[-1][:last_idx]
            return b''.join(total_recved), remained

    def close(self):
        '''Close the socket and release resources'''
        if self.isopen:
            self.isopen = False
            self.sock.close()
            self.buffsize = 0


class CLIENT(object):
    '''Client for transfer'''

    DATA_TYPES = ['data', 'signal']
    USED_PORT = set() # Set that used to save used port numbers.

    def __init__(self, bind_addr=None, backlog=10, islocal=False, IPv=4, isTCP=True, reuse_port=True,
                 timeout=None, sock_blocking=True, buffsize=2048, max_send_qsize=100, max_recv_qsize=100):
        '''
        Init client for communication.

        Args:
            bind_addr:      tuple (ip, port). Default is None. If bind_addr is None, the client will use (current_external_ip, port=7000)
                            as recv addr. If 
            backlog:        int. Default is 10. Integer that specifies number of connection allowed before acceptance.
            islocal:        bool. Default is False. Indicating the socket type is local or internet. Note that local socket is not
                            supported on Win patforms. Thus, it is ignored on Win patforms.
            IPv:            int. Default is 4. Integer that indicating version of IP protocol. Only IPv4 or IPv6 supported.
            isTCP:          bool. Default is True. Bool that specifies socket type: TCP (for True) or UDP (for False).
            reuse_port:     bool. Default is True. Bool that indicating whther reuse the bind port (recv related port).
            timeout:        None or (float, int) >= 0. Default is None. None means generate blocking socket and keep blocked until data
                            is ready. Value of timeout (float, int) > 0 means keep blocked until data is ready or time limit exceeded.
                            Value of timeout (float, int) == 0 means Non-blocking socket.
            sock_blocking:  bool. Default is True. True value is exactly the same as 'timeout=None', where socket will keep blocked forever
                            unless data is ready. False value is exactly the same as 'timeout=0', where generate non-blocking socket.
            buffsize:       int. Default is 2048. Specifies the size of recv buffer in bytes.
            max_send_qsize: int. Default is 100. Specifies the max number of obj in the send queue.
            max_recv_qsize: int. Default is 100. Specifies the max number of obj in the recv queue.
        '''
        # init resources
        self.selector = selectors.DefaultSelector()
        self.send_sock_processor = SOCKET_PROCESSOR(islocal, IPv, isTCP, sock_blocking, timeout, buffsize)
        self.recv_sock_processor = SOCKET_PROCESSOR(islocal, IPv, isTCP, sock_blocking, timeout, buffsize)

        # ip and port initialization
        self.pub_ip = _get_current_host_ip()
        self.bind(bind_addr)

        # ansync data transfer
        self.send_queue = multiprocessing.Queue(max_send_qsize)
        self.recv_queue = multiprocessing.Queue(max_recv_qsize)
        self.send_thread = threading.Thread(target=self._send_loop, name='Send Thread', args=(self.send_queue,), daemon=True)
        self.recv_thread = threading.Thread(target=self._recv_loop, name='Recv Thread', args=(self.recv_queue,), daemon=True)

        # start listening
        self.isrunning = False
        self.backlog = backlog
        self.run()

    def bind(self, bind_addr):
        if bind_addr is None:
            bind_addr = self.pub_ip, 7000
        ip, port = bind_addr
        ip, port = (ip if ip else self.pub_ip), _find_N_free_ports(*bind_addr, N=1, used_ports_set=self.USED_PORT)[0]
        self.USED_PORT.add(port)
        bind_addr = ip, port
        self.recv_sock_processor.bind(bind_addr)
        self.bind_addr = bind_addr
        return self.bind_addr

    def connect(self, server_addr):
        '''Connect to given Server beore send data'''
        self.send_sock_processor.connect_to_new_addr(server_addr)

    def accept(self, sock, mask):
        '''Should be ready'''
        # accept use new socket representing a connection
        new_sock, addr = sock.accept()
        new_sock.setblocking(False)
        self.selector.register(new_sock, selectors.EVENT_READ, self._recv_to_queue)
        return new_sock, addr

    def _send_loop(self, send_queue):
        while self.isrunning:
            try:
                data, type, encoding = send_queue.get(timeout=0.01)
                self.send_sock_processor.send(data, type, encoding)
            except QEmptyException:
                continue

    def _recv_loop(self, recv_queue):
        while self.isrunning:
            for k, mask in self.selector.select(timeout=0.01):
                callback = k.data
                callback(k.fileobj, mask)

    def send(self, data, type='data', encoding='utf8'):
        '''Send data to server. Return send bytes cnt.'''
        assert type in self.DATA_TYPES, "Not suported data type description:'%s'. Only %r supported!." % (type(data), self.DATA_TYPES)
        self.send_queue.put((data, type, encoding))
    
    def _recv_to_queue(self, sock, mask):
        '''Save received data from server to recv_queue.'''
        recved = self.recv_sock_processor.recv(sock, mask)
        if recved:
            self.recv_queue.put(recved)
    
    def recv(self, timeout=None):
        '''Receive data from server. Return tuple: (data_type, data)'''
        if timeout:
            return self.recv_queue.get(timeout=timeout)
        else:
            return self.recv_queue.get()
    
    def run(self):
        if not self.isrunning:
            self.isrunning = True
            self.recv_sock_processor.listen(self.backlog)
            self.selector.register(self.recv_sock_processor.sock, selectors.EVENT_READ, self.accept)
            self.send_thread.start()
            self.recv_thread.start()
    
    def close(self):
        if self.isrunning:
            self.isrunning = False
            self.USED_PORT.discard(self.bind_addr[1])
            self.selector.close()
            self.recv_thread.join()          
            self.send_thread.join()
            self.send_sock_processor.close()
            self.recv_sock_processor.close()
            self.send_queue.close()
            self.recv_queue.close()
    
    def __del__(self):
        self.close()


if __name__ == '__main__':
    # 测试本地回环地址通信情况
    addr1, addr2 = ('localhost', 1234), ('localhost', 1234)
    # 测试本地外网地址通信情况, 默认接收端口为 7000
    addr1, addr2, recv_port1, recv_port2 = None, None, 7000, 7000
    ip_addr = _get_current_host_ip()
    # 测试自动寻找可用的空闲端口, 默认最小端口为7000, 如果端口已被占用, 则自动替换为7000以上的可用端口
    c1 = CLIENT(bind_addr=addr1)
    c2 = CLIENT(bind_addr=addr2)
    print('c1 bind addr:', c1.bind_addr)
    print('c2 bind addr:', c2.bind_addr)
    c1.connect(c2.bind_addr)
    c2.connect(c1.bind_addr)
    c1.send(data='你好啊客户端2, 我是客户端1')
    c2.send(data='你好啊客户端1, 我是客户端2')
    
    print(c1.recv())
    print(c2.recv())
    c1.close()
    c2.close()
    print(ip_addr)

    print('exit')
