

class Metrics:
    """
    Compute inter-rater metrics 
    """
    def __init__(self):
        self.observed= None
        self.expected= None

    def kappa(self):
        """
        Compute Cohen's Kappa
        """
        kappa = (self.observed - self.expected) / (1 - self.expected)
        return "In progress"
    

class K_p(Metrics):
    def __init__(self):
        super().__init__()
        self.compute_cobsp()
        self.compute_cexpp()

    def compute_cobsp(self):
        self.observed = 0.8

    def compute_cexpp(self):
        self.expected = 0.6