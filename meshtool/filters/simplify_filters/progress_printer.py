import sys

class ProgressPrinter:
    def __init__(self, total):
        self.total = total
        self.i = 0
        self.last_pct = -1

    def step(self):
        pct = 100*self.i/self.total
        if pct != self.last_pct:
            sys.stdout.write(str(pct)+"%\r")
            sys.stdout.flush()
            self.last_pct = pct
        self.i += 1
