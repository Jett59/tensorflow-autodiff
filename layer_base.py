class Layer:
    def __init__(self):
        # We want to make sure the subclass implemented the `calculate`, `variables` `build` methods.
        assert hasattr(self, "calculate")
        assert hasattr(self, "variables")
        assert hasattr(self, "build")
