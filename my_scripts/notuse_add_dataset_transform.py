class AddDatasetName:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, atoms):
        atoms.info["dataset"] = self.name
        return atoms
