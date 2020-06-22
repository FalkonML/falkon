class FakeQueue:
    def __init__(self):
        self.lst = []

    def get(self):
        return self.lst.pop(0)

    def put(self, obj):
        self.lst.append(obj)

    def __len__(self):
        return len(self.lst)