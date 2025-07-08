import torch
import json
from datetime import datetime
from torchvision.transforms import v2

from pathlib import Path
import logging
from typing import Dict, List
from transformers import AutoTokenizer

from experiments.metrics import RougeScore, BleuScore, PerplexityMetric
from inference.inference import load_model
from experiments.loaders import get_dataloaders


class ModelEvaluator:
    def __init__(self, model_path: str, data_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = load_model(model_path, device)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

        transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _, _, self.test_loader = get_dataloaders(data_path, batch_size=6, transform=transform)

        self.rouge_scorer = RougeScore()
        self.bleu_scorer = BleuScore()
        self.perplexity_scorer = PerplexityMetric()

        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('logs/evaluation')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'evaluation_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.metrics_file = log_dir / f'metrics_{timestamp}.json'

    def calculate_metrics(self, reference: str, candidate: str, probabilities: List[float]) -> Dict:
        """Calculate all metrics for a single prediction"""
        reference = reference if isinstance(reference, str) else self.tokenizer.decode(reference, skip_special_tokens=True)
        candidate = candidate if isinstance(candidate, str) else self.tokenizer.decode(candidate, skip_special_tokens=True)

        rouge_1 = self.rouge_scorer.rouge_n(reference, candidate, n=1)
        rouge_2 = self.rouge_scorer.rouge_n(reference, candidate, n=2)
        rouge_l = self.rouge_scorer.rouge_l(reference, candidate)

        bleu = self.bleu_scorer.calculate(reference, candidate)

        perplexity = self.perplexity_scorer.calculate(probabilities)

        return {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'bleu': bleu,
            'perplexity': perplexity
        }

    def evaluate(self) -> Dict:
        """Evaluate the model on the test set"""
        self.logger.info("Starting model evaluation...")
        
        all_metrics = []
        total_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)

                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=-1)

                for idx in range(len(images)):
                    reference = self.tokenizer.decode(labels[idx], skip_special_tokens=True)
                    top_prob, top_idx = probabilities[idx].max(0)
                    candidate = self.tokenizer.decode([top_idx.item()], skip_special_tokens=True)

                    sample_metrics = self.calculate_metrics(
                        reference,
                        candidate,
                        probabilities[idx].cpu().tolist()
                    )
                    all_metrics.append(sample_metrics)

                    self.logger.debug(f"Sample {total_samples + idx + 1}:")
                    self.logger.debug(f"Reference: {reference}")
                    self.logger.debug(f"Prediction: {candidate}")
                    self.logger.debug(f"Metrics: {sample_metrics}")

                total_samples += len(images)

                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed {total_samples} samples...")

        avg_metrics = self.calculate_average_metrics(all_metrics)

        self.logger.info("Evaluation completed!")
        self.logger.info(f"Total samples evaluated: {total_samples}")
        self.logger.info("Final metrics:")
        self.logger.info(json.dumps(avg_metrics, indent=2))

        metrics_data = {
            'average_metrics': avg_metrics,
            'individual_samples': all_metrics,
            'total_samples': total_samples,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.metrics_file, 'w') as file:
            json.dump(metrics_data, file, indent=2)

        return avg_metrics

    def calculate_average_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Calculate average metrics across all samples"""
        avg_metrics = {
            'rouge_1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rouge_2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rouge_l': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'bleu': 0.0,
            'perplexity': 0.0
        }

        n_samples = len(all_metrics)

        for metrics in all_metrics:
            for rouge_type in ['rouge_1', 'rouge_2', 'rouge_l']:
                for key in ['precision', 'recall', 'f1']:
                    avg_metrics[rouge_type][key] += metrics[rouge_type][key] / n_samples

            avg_metrics['bleu'] += metrics['bleu'] / n_samples
            avg_metrics['perplexity'] += metrics['perplexity'] / n_samples

        return avg_metrics

def main():
    model_path = 'logs/final/1st run/models/final_model.pth'
    data_path = 'data'

    evaluator = ModelEvaluator(model_path, data_path)
    metrics = evaluator.evaluate()

    print("\nEvaluation completed! Check the logs directory for detailed results.")

if __name__ == "__main__":
    main()
