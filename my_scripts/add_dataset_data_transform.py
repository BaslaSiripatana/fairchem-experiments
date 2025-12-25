class AddDatasetToAtomicData:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, data):
        data.dataset = [self.name]
        return data
