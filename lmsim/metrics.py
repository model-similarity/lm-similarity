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
    

class EC(Metrics):
    def __init__(self):
        super().__init__()
        """
        Compute Error Consistency
        """
        self.acc_a = None
        self.acc_b = None

    def compute_cobs(self, output_a, output_b, gt):
        gt = np.array(gt)
        preds_a = np.array([np.argmax(a) for a in output_a])
        preds_b = np.array([np.argmax(b) for b in output_b])

        # Compare each prediction with ground-truth
        correct_a = (preds_a == gt)
        correct_b = (preds_b == gt)

        # "cobs" = how many times the two predictions match in correctness (both right or both wrong)
        cobs = np.sum(correct_a == correct_b)
      
        self.observed = cobs/len(output_a)
        self.acc_a = np.sum(correct_a)/len(output_a)
        self.acc_b = np.sum(correct_b)/len(output_b)

    def compute_cexp(self):
        cexp = self.acc_a * self.acc_b + (1-self.acc_a)*(1-self.acc_b)
        self.expected = cexp
    
    def compute_k(self, output_a:list[np.array], output_b:list[np.array], gt:list[int])->float:
        """
        Compute error consistency
        input:
        prob_a: list of softmax probabilities (np.array) or one-hot vector (np.array) for model A
        prob_b: list of softmax probabilities (np.array) or one-hot vector (np.array) for model B
        gt: list of ground truth index (int)
        output:
        kappa: similairty (float)
        """
        self.compute_cobs(output_a, output_b, gt)
        self.compute_cexp()
        return self.kappa()

    

class Goels_k(Metrics):
    def __init__(self, prob:bool=True):
        super().__init__()
        """
        Compute Goels $k$
        Default: 
        - prob=True, compute Goels $k_p$ based on softmax probability
        - prob=False, compute Goels $k$ based on one-hot vector (discrete)
        """
        self.p_hat_a = None
        self.p_hat_b = None
        self.frac = None
        self.prob = prob

    def compute_cobsp(self, output_a, output_b):
        cobsp = 0
        for sample_a, sample_b in zip(output_a, output_b):
            assert len(sample_a) == len(sample_b), "Model ouput must be equal length"
            if self.prob:
                cobsp += np.sum(sample_a * sample_b)
            else:
                cobsp += int(sum(abs(sample_a - sample_b)) == 0)
      
        self.observed = cobsp/len(output_a)

    def compute_phat(self,output_a, output_b, gt):
        phat_a = 0
        phat_b = 0
        for idx, (sample_a, sample_b) in enumerate(zip(output_a, output_b)):
            assert gt[idx] < len(sample_a), "Ground truth index must be in range of the number of option in a sample"
            phat_a += sample_a[gt[idx]]
            phat_b += sample_b[gt[idx]]
            

        self.p_hat_a = phat_a/len(output_a)
        self.p_hat_b = phat_b/len(output_b)

    def compute_frac(self, output_a):
        frac = 0
        for sample in output_a:
            frac += 1/(len(sample)-1)
        self.frac = frac/len(output_a)

    def compute_cexpp(self):
        cexp = self.p_hat_a * self.p_hat_b + self.frac * (1-self.p_hat_a )*(1-self.p_hat_b)
        self.expected = cexp
    
    def compute_k(self, output_a:list[np.array], output_b:list[np.array], gt:list[int])->float:
        """
        Compute Goels $k$
        input:
        prob_a: list of softmax probabilities (np.array) or one-hot vector (np.array) for model A
        prob_b: list of softmax probabilities (np.array) or one-hot vector (np.array) for model B
        gt: list of ground truth index (int)
        output:
        kappa: similairty (float)
        """
        self.compute_cobsp(output_a, output_b)
        self.compute_phat(output_a, output_b, gt)
        self.compute_frac(output_a)
        self.compute_cexpp()
        return self.kappa()

