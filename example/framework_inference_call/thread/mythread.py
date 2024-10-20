import threading

class MyThread(threading.Thread):
    def __init__(self, thread_id, thread_name) -> None:
        self.id = thread_id
        self.name = thread_name
    
    def run(self):
        print("Starting ", self.name)
        print("Exiting ", self.name)
        