"""
Evaluation metrics for video captioning.

This module implements standard captioning evaluation metrics used in the field:
- BLEU (Bilingual Evaluation Understudy): N-gram precision with brevity penalty
- CIDEr (Consensus-based Image Description Evaluation): TF-IDF weighted n-gram consensus
- METEOR (Metric for Evaluation of Translation with Explicit ORdering): Alignment-based with synonyms
- ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)

All metrics return scores between 0 and 1, where 1 indicates perfect match.

Example Usage:
    >>> from src.evaluation.metrics import MetricsCalculator
    >>> 
    >>> calculator = MetricsCalculator()
    >>> 
    >>> predictions = ["the cat sat on the mat"]
    >>> references = [["a cat is sitting on a mat"]]
    >>> 
    >>> results = calculator.compute_all(predictions, references)
    >>> print(f"BLEU-4: {results['bleu_4']:.4f}")
    >>> print(f"CIDEr: {results['cider']:.4f}")
"""

import nltk
from typing import List, Dict, Union, Optional
import numpy as np
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class BLEUScorer:
    """
    BLEU (Bilingual Evaluation Understudy) scorer.
    Computes BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.
    """
    
    def __init__(self):
        """Initialize BLEU scorer."""
        pass
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - can be enhanced
        text = text.lower()
        # Remove punctuation but keep apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        return text.split()
    
    def _compute_bleu_n(self, prediction: str, references: List[str], n: int) -> float:
        """
        Compute BLEU-n score for a single prediction.
        
        Args:
            prediction: Predicted caption
            references: List of reference captions
            n: N-gram order (1-4)
            
        Returns:
            BLEU-n score (0-1)
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens_list = [self._tokenize(ref) for ref in references]
        
        if len(pred_tokens) < n:
            return 0.0
        
        # Step 1: Extract n-grams from prediction
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        
        # Step 2: Count maximum matches across all references
        # For each n-gram in prediction, find the maximum count across all references
        # This handles multiple reference captions (takes the best match)
        max_counts = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            for ngram in pred_ngrams:
                # Keep maximum count across all references (best-case matching)
                max_counts[ngram] = max(max_counts[ngram], ref_ngrams[ngram])
        
        # Step 3: Calculate n-gram precision
        # Count how many n-grams in prediction match references (clipped to reference count)
        # Clipping prevents over-counting when prediction has more occurrences than reference
        matches = sum(min(pred_ngrams[ngram], max_counts[ngram]) for ngram in pred_ngrams)
        total = sum(pred_ngrams.values())
        
        if total == 0:
            return 0.0
        
        precision = matches / total  # Precision = matched n-grams / total n-grams
        
        # Step 4: Apply brevity penalty
        # Penalizes predictions that are shorter than the reference
        # This prevents the model from generating very short captions to maximize precision
        pred_len = len(pred_tokens)
        ref_lens = [len(ref) for ref in ref_tokens_list]
        # Find reference length closest to prediction length
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - pred_len))
        
        # Brevity penalty: 1.0 if prediction is longer, exponential decay if shorter
        if pred_len > closest_ref_len:
            bp = 1.0  # No penalty if prediction is longer than reference
        else:
            # Exponential penalty: exp(1 - ref_len / pred_len)
            # Shorter predictions get penalized more
            bp = np.exp(1 - closest_ref_len / pred_len) if pred_len > 0 else 0.0
        
        return bp * precision
    
    def compute(self, predictions: Union[str, List[str]], 
                references: Union[str, List[str], List[List[str]]]) -> Dict[str, float]:
        """
        Compute BLEU-1 through BLEU-4 scores.
        
        Args:
            predictions: Single prediction string or list of predictions
            references: Single reference string, list of references, or list of reference lists
            
        Returns:
            Dictionary with 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4' scores
        """
        # Normalize inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        # Compute corpus-level BLEU scores
        bleu_scores = {f'bleu_{i}': [] for i in range(1, 5)}
        
        for pred, refs in zip(predictions, references):
            for n in range(1, 5):
                score = self._compute_bleu_n(pred, refs, n)
                bleu_scores[f'bleu_{n}'].append(score)
        
        # Average across all predictions
        result = {key: np.mean(scores) for key, scores in bleu_scores.items()}
        
        return result


class CIDErScorer:
    """
    CIDEr (Consensus-based Image Description Evaluation) scorer.
    Uses pycocoevalcap if available, otherwise implements simplified version.
    """
    
    def __init__(self):
        """Initialize CIDEr scorer."""
        try:
            from pycocoevalcap.cider.cider import Cider
            self.use_pycoco = True
            self.cider = Cider()
        except ImportError:
            self.use_pycoco = False
    
    def compute(self, predictions: Union[str, List[str]], 
                references: Union[str, List[str], List[List[str]]]) -> Dict[str, float]:
        """
        Compute CIDEr score.
        
        Args:
            predictions: Single prediction string or list of predictions
            references: Single reference string, list of references, or list of reference lists
            
        Returns:
            Dictionary with 'cider' score
        """
        # Normalize inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        if self.use_pycoco:
            # Use pycocoevalcap
            gts = {i: refs for i, refs in enumerate(references)}
            res = {i: [pred] for i, pred in enumerate(predictions)}
            score, _ = self.cider.compute_score(gts, res)
            return {'cider': float(score)}
        else:
            # Simplified CIDEr implementation
            # This is a basic version - full CIDEr requires TF-IDF weighting
            scores = []
            for pred, refs in zip(predictions, references):
                # Tokenize
                pred_tokens = pred.lower().split()
                ref_tokens_list = [ref.lower().split() for ref in refs]
                
                # Compute n-gram matches (simplified)
                matches = 0
                total = len(pred_tokens)
                
                for ref_tokens in ref_tokens_list:
                    ref_set = set(ref_tokens)
                    matches += sum(1 for token in pred_tokens if token in ref_set)
                
                if total > 0:
                    score = matches / (total * len(refs))
                else:
                    score = 0.0
                
                scores.append(score)
            
            return {'cider': np.mean(scores)}


class METEORScorer:
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) scorer.
    """
    
    def __init__(self):
        """Initialize METEOR scorer."""
        try:
            from nltk.translate.meteor_score import meteor_score
            self.meteor_score = meteor_score
        except ImportError:
            self.meteor_score = None
    
    def compute(self, predictions: Union[str, List[str]], 
                references: Union[str, List[str], List[List[str]]]) -> Dict[str, float]:
        """
        Compute METEOR score.
        
        Args:
            predictions: Single prediction string or list of predictions
            references: Single reference string, list of references, or list of reference lists
            
        Returns:
            Dictionary with 'meteor' score
        """
        if self.meteor_score is None:
            # Fallback: return 0 if METEOR not available
            return {'meteor': 0.0}
        
        # Normalize inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        scores = []
        for pred, refs in zip(predictions, references):
            # METEOR typically uses the best reference
            best_score = 0.0
            for ref in refs:
                try:
                    score = self.meteor_score([ref.split()], pred.split())
                    best_score = max(best_score, score)
                except Exception:
                    pass
            scores.append(best_score)
        
        return {'meteor': np.mean(scores) if scores else 0.0}


class ROUGEScorer:
    """
    ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest) scorer.
    """
    
    def __init__(self):
        """Initialize ROUGE scorer."""
        try:
            from rouge_score import rouge_scorer
            self.use_rouge_package = True
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except ImportError:
            self.use_rouge_package = False
    
    def _lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _compute_rouge_l(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-L score for a single pair."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            return {'rouge_l_f1': 0.0, 'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0}
        
        lcs_len = self._lcs(pred_tokens, ref_tokens)
        
        precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'rouge_l_f1': f1,
            'rouge_l_precision': precision,
            'rouge_l_recall': recall
        }
    
    def compute(self, predictions: Union[str, List[str]], 
                references: Union[str, List[str], List[List[str]]]) -> Dict[str, float]:
        """
        Compute ROUGE-L score.
        
        Args:
            predictions: Single prediction string or list of predictions
            references: Single reference string, list of references, or list of reference lists
            
        Returns:
            Dictionary with 'rouge_l_f1', 'rouge_l_precision', 'rouge_l_recall' scores
        """
        # Normalize inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        if self.use_rouge_package:
            # Use rouge-score package
            f1_scores = []
            precision_scores = []
            recall_scores = []
            
            for pred, refs in zip(predictions, references):
                best_f1 = 0.0
                best_precision = 0.0
                best_recall = 0.0
                
                for ref in refs:
                    scores = self.scorer.score(ref, pred)
                    rouge_l = scores['rougeL']
                    best_f1 = max(best_f1, rouge_l.fmeasure)
                    best_precision = max(best_precision, rouge_l.precision)
                    best_recall = max(best_recall, rouge_l.recall)
                
                f1_scores.append(best_f1)
                precision_scores.append(best_precision)
                recall_scores.append(best_recall)
            
            return {
                'rouge_l_f1': np.mean(f1_scores),
                'rouge_l_precision': np.mean(precision_scores),
                'rouge_l_recall': np.mean(recall_scores)
            }
        else:
            # Use custom implementation
            all_scores = []
            for pred, refs in zip(predictions, references):
                best_score = {'rouge_l_f1': 0.0, 'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0}
                for ref in refs:
                    score = self._compute_rouge_l(pred, ref)
                    if score['rouge_l_f1'] > best_score['rouge_l_f1']:
                        best_score = score
                all_scores.append(best_score)
            
            return {
                'rouge_l_f1': np.mean([s['rouge_l_f1'] for s in all_scores]),
                'rouge_l_precision': np.mean([s['rouge_l_precision'] for s in all_scores]),
                'rouge_l_recall': np.mean([s['rouge_l_recall'] for s in all_scores])
            }


class MetricsCalculator:
    """
    Unified metrics calculator that computes all evaluation metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator with all scorers."""
        self.bleu_scorer = BLEUScorer()
        self.cider_scorer = CIDErScorer()
        self.meteor_scorer = METEORScorer()
        self.rouge_scorer = ROUGEScorer()
    
    def compute_all(self, predictions: Union[str, List[str]], 
                   references: Union[str, List[str], List[List[str]]]) -> Dict[str, float]:
        """
        Compute all metrics and return comprehensive results.
        
        Args:
            predictions: Single prediction string or list of predictions
            references: Single reference string, list of references, or list of reference lists
            
        Returns:
            Dictionary with all metric scores:
            - bleu_1, bleu_2, bleu_3, bleu_4
            - cider
            - meteor
            - rouge_l_f1, rouge_l_precision, rouge_l_recall
        """
        results = {}
        
        # Compute BLEU scores
        bleu_results = self.bleu_scorer.compute(predictions, references)
        results.update(bleu_results)
        
        # Compute CIDEr score
        cider_results = self.cider_scorer.compute(predictions, references)
        results.update(cider_results)
        
        # Compute METEOR score
        meteor_results = self.meteor_scorer.compute(predictions, references)
        results.update(meteor_results)
        
        # Compute ROUGE-L scores
        rouge_results = self.rouge_scorer.compute(predictions, references)
        results.update(rouge_results)
        
        return results
    
    def compute_batch(self, predictions: List[str], 
                     references: List[Union[str, List[str]]]) -> Dict[str, float]:
        """
        Compute metrics for a batch of predictions.
        
        Args:
            predictions: List of prediction strings
            references: List of reference strings or list of reference lists
            
        Returns:
            Dictionary with all metric scores (averaged across batch)
        """
        return self.compute_all(predictions, references)

