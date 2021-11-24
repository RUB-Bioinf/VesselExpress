# Weilin Fu (2019) Frangi-Net on High-Resolution Fundus (HRF) image database [Source Code].
# https://doi.org/10.24433/CO.5016803.v2
# GNU General Public License (GPL)

import threading


class GraphComputationThread(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.lock.acquire()
        self.res = None
        self.session = None
        self.operations = None
        self.feed_dict = None
        self.thread = None
        self.lock.release()

    def setSession(self, session):
        self.lock.acquire()
        self.session = session
        self.lock.release()

    def setParameters(self, operations, feed_dict):
        self.lock.acquire()
        self.operations = operations
        self.feed_dict = feed_dict
        self.lock.release()

    def start(self):
        if self.thread is not None:
            raise Exception('join() needs to be called before restarting thread')
        self.thread = threading.Thread(target=self)
        self.thread.start()

    def __call__(self):
        self.lock.acquire()
        self.res = self.session.run(self.operations, feed_dict=self.feed_dict)
        self.lock.release()

    def join(self):
        self.thread.join()
        self.thread = None

    def getResult(self):
        self.lock.acquire()
        res = self.res
        self.lock.release()
        return res