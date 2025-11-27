import logging
from typing import Dict, Optional

import joblib
from core.config.settings import settings

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            model_path = settings.model_full_path

            if not model_path.exists():
                logger.warning(f"Modelo não encontrado em: {model_path}")
                logger.warning("Usando modo mock. Coloque o modelo em: data/traits_pipeline.joblib")
                return

            logger.info(f"Carregando modelo de: {model_path}")
            self.model = joblib.load(model_path)
            logger.info("Modelo carregado com sucesso!")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            logger.warning("Continuando em modo mock")
            self.model = None

    def predict(self, text: str) -> Dict[str, any]:
        if self.model is None:
            return self._mock_prediction(text)

        try:
            predictions = self.model.predict_proba([text])[0]

            if hasattr(self.model, 'classes_'):
                labels = self.model.classes_
            elif hasattr(self.model, 'named_steps') and hasattr(
                    self.model.named_steps.get('classifier', None),
                    'classes_'
            ):
                labels = self.model.named_steps['classifier'].classes_
            else:
                raise AttributeError("Não foi possível encontrar as classes do modelo")

            scores_dict = {label: float(score) for label, score in zip(labels, predictions)}
            sorted_labels = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            significant_labels = [label for label, score in sorted_labels if score > 0.1][:5]

            if not significant_labels:
                significant_labels = [label for label, _ in sorted_labels[:2]]

            result = {
                "labels": significant_labels,
                "scores": {label: scores_dict[label] for label in significant_labels}
            }

            logger.info(f"Predição realizada: {result['labels']}")
            return result

        except Exception as e:
            logger.error(f"Erro durante predição: {e}")
            return self._mock_prediction(text)

    def _mock_prediction(self, text: str) -> Dict[str, any]:
        logger.info("Usando predição mock")
        return {
            "labels": ["protetor", "determinado"],
            "scores": {
                "protetor": 0.7141908960625957,
                "determinado": 0.2183861662956591
            }
        }

    def get_model_info(self) -> Dict[str, any]:
        return {
            "model_loaded": self.model is not None,
            "model_path": str(settings.model_full_path),
            "model_exists": settings.model_full_path.exists()
        }


_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
