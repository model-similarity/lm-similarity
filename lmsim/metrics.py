import numpy as np

class Metrics:
    """
    Compute inter-rater metrics 
    """
    def __init__(self):
        self.observed= None
        self.expected= None

    def kappa(self):
        """
        Compute Kappa
        """
        kappa = (self.observed - self.expected) / (1 - self.expected)
        return kappa
    

class K_p(Metrics):
    def __init__(self):
        super().__init__()
        """
        Compute probabilistic error consistency
        """
        self.p_hat_a = None
        self.p_hat_b = None
        self.frac = None

    def compute_cobsp(self, prob_a, prob_b):
        cobsp = 0
        for sample_a, sample_b in zip(prob_a, prob_b):
            assert len(sample_a) == len(sample_b), "Ouput probabilities must be equal length"
            cobsp += np.sum(sample_a * sample_b)
      
        
        self.observed = cobsp/len(prob_a)

    def compute_phat(self,prob_a, prob_b, gt):
        phat_a = 0
        phat_b = 0
        for idx, (sample_a, sample_b) in enumerate(zip(prob_a, prob_b)):
            phat_a += sample_a[gt[idx]]
            phat_b += sample_b[gt[idx]]

        self.p_hat_a = phat_a/len(prob_a)
        self.p_hat_b = phat_b/len(prob_b)

    def compute_frac(self, prob_a):
        frac = 0
        for sample in prob_a:
            frac += 1/(len(sample)-1)
        self.frac = frac/len(prob_a)

    def compute_cexpp(self):
        cexp = self.p_hat_a * self.p_hat_b + self.frac * (1-self.p_hat_a )*(1-self.p_hat_b)
        self.expected = cexp
    
    def compute_kp(self, prob_a:list[np.array], prob_b:list[np.array], gt:list[int])->float:
        """
        Compute probabilistic error consistency
        input:
        prob_a: prob of rater A
        prob_b: prob of rater B
        gt: ground truth index
        output:
        k_p: probabilistic error consistency
        """
        self.compute_cobsp(prob_a, prob_b)
        self.compute_phat(prob_a, prob_b, gt)
        self.compute_frac(prob_a)
        self.compute_cexpp()
        return self.kappa()

