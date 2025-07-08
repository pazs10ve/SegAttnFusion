from collections import Counter
from typing import List, Dict, Union, Tuple
import numpy as np
import math
from nltk.util import ngrams

class RougeScore:
    def __init__(self):
        self.beta = 1.2  # For ROUGE-L F1 calculation
    
    def _get_ngrams(self, n: int, text: str) -> Counter:
        """Convert text into character n-grams and count them."""
        print("text = ", text)
        tokens = text.lower().split()
        return Counter(ngrams(tokens, n))
    
    def rouge_n(self, reference: str, candidate: str, n: int = 1) -> Dict[str, float]:
        """
        Calculate ROUGE-N score.
        Args:
            reference: Reference text
            candidate: Candidate text
            n: n-gram length
        Returns:
            Dictionary containing precision, recall, and f1 scores
        """
        ref_ngrams = self._get_ngrams(n, reference)
        cand_ngrams = self._get_ngrams(n, candidate)
        
        # Calculate overlapping n-grams
        overlap = sum((ref_ngrams & cand_ngrams).values())
        
        # Calculate precision and recall
        precision = overlap / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0
        recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def rouge_l(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE-L score using longest common subsequence.
        """
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)
        
        # Calculate precision and recall
        precision = lcs_length / len(cand_tokens) if len(cand_tokens) > 0 else 0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        # Calculate F1 score with beta
        if precision + recall > 0:
            f1 = ((1 + self.beta ** 2) * precision * recall) / \
                 (recall + (self.beta ** 2) * precision)
        else:
            f1 = 0
            
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _lcs_length(self, ref_tokens: List[str], cand_tokens: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

class BleuScore:
    def __init__(self, weights: List[float] = None):
        """
        Initialize BLEU score calculator.
        Args:
            weights: Weights for n-grams (default: [0.25, 0.25, 0.25, 0.25] for BLEU-4)
        """
        self.weights = weights if weights else [0.25, 0.25, 0.25, 0.25]
        
    def calculate(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score.
        Args:
            reference: Reference text
            candidate: Candidate text
        Returns:
            BLEU score
        """
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        # Calculate brevity penalty
        bp = self._brevity_penalty(len(ref_tokens), len(cand_tokens))
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, len(self.weights) + 1):
            precision = self._modified_precision(ref_tokens, cand_tokens, n)
            precisions.append(precision)
        
        # Calculate final score
        if min(precisions) > 0:
            log_precisions = [w * math.log(p) for w, p in zip(self.weights, precisions)]
            score = bp * math.exp(sum(log_precisions))
        else:
            score = 0.0
            
        return score
    
    def _modified_precision(self, ref_tokens: List[str], cand_tokens: List[str], n: int) -> float:
        """Calculate modified n-gram precision."""
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        cand_ngrams = Counter(ngrams(cand_tokens, n))
        
        total_matches = sum((ref_ngrams & cand_ngrams).values())
        total_candidates = max(sum(cand_ngrams.values()), 1)
        
        return total_matches / total_candidates
    
    def _brevity_penalty(self, ref_length: int, cand_length: int) -> float:
        """Calculate brevity penalty."""
        if cand_length > ref_length:
            return 1
        elif cand_length == 0:
            return 0
        else:
            return math.exp(1 - ref_length / cand_length)

class PerplexityMetric:
    def __init__(self):
        self.epsilon = 1e-10  # Small constant to avoid log(0)
    
    def calculate(self, probabilities: List[float]) -> float:
        """
        Calculate perplexity given token probabilities.
        Args:
            probabilities: List of token probabilities
        Returns:
            Perplexity score
        """
        # Add epsilon to avoid log(0)
        safe_probs = [max(p, self.epsilon) for p in probabilities]
        
        # Calculate log probabilities
        log_probs = [math.log2(p) for p in safe_probs]
        
        # Calculate average negative log probability
        avg_neg_log_prob = -sum(log_probs) / len(log_probs)
        
        # Calculate perplexity
        perplexity = math.pow(2, avg_neg_log_prob)
        
        return perplexity
    
    def calculate_from_sentences(self, sentences: List[str], model_probs: List[List[float]]) -> float:
        """
        Calculate perplexity for multiple sentences.
        Args:
            sentences: List of input sentences
            model_probs: List of probability lists for each sentence
        Returns:
            Average perplexity across all sentences
        """
        perplexities = []
        for probs in model_probs:
            perplexity = self.calculate(probs)
            perplexities.append(perplexity)
            
        return sum(perplexities) / len(perplexities)
    



"""
    # ROUGE score example
rouge = RougeScore()
reference = "the cat sat on the mat"
candidate = "the cat was on the mat"
rouge_1_scores = rouge.rouge_n(reference, candidate, n=1)
rouge_l_scores = rouge.rouge_l(reference, candidate)

# BLEU score example
bleu = BleuScore()
bleu_score = bleu.calculate(reference, candidate)

# Perplexity example
perplexity = PerplexityMetric()
token_probs = [0.2, 0.1, 0.3, 0.4]
perplexity_score = perplexity.calculate(token_probs)


print(rouge_1_scores)
print(rouge_l_scores)
print(bleu_score)
print(perplexity_score)
"""