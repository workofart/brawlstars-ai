import random, numpy as np

class Experience_Buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        # current size almost exceeding buffer size
        if len(self.buffer) + 1 >= self.buffer_size: 
            self.buffer[0:(1+len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size):
        point = np.random.randint(0,len(self.buffer)+1-batch_size)
        return self.buffer[point:point+batch_size]

    def clear(self):
        self.buffer = []
    
    def size(self):
        return len(self.buffer)