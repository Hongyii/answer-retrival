class Question:
    def __init__(self, data):
        self.answerId = data[0]
        self.body = data[1]
        self.Id = data[2]
        self.postTypeId = 1
        self.tags = data[3]
        self.title = data[4]

