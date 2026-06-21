"""
Model registry for managing and instantiating captioning models.
"""

from typing import Dict, Type, Optional, Any, List
from .base_model import BaseCaptionModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Registry for all captioning models.
    Allows easy model instantiation and management.
    """
    
    _models: Dict[str, Type[BaseCaptionModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseCaptionModel]):
        """
        Register a model class.
        
        Args:
            name: Model name identifier
            model_class: Model class that inherits from BaseCaptionModel
        """
        if not issubclass(model_class, BaseCaptionModel):
            raise ValueError(f"Model class must inherit from BaseCaptionModel")
        
        cls._models[name] = model_class
        logger.info(f"Registered model: {name} ({model_class.__name__})")
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseCaptionModel:
        """
        Create an instance of a registered model.
        
        Args:
            name: Model name identifier
            config: Optional configuration dictionary
            
        Returns:
            Model instance
        """
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        
        model_class = cls._models[name]
        # Pass config as config_dict to match model signatures
        return model_class(config_dict=config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseCaptionModel]:
        """
        Get the model class for a given name.
        
        Args:
            name: Model name identifier
            
        Returns:
            Model class
        """
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        
        return cls._models[name]


# Convenience functions
def register_model(name: str, model_class: Type[BaseCaptionModel]):
    """Register a model."""
    ModelRegistry.register(name, model_class)


def create_model(name: str, config: Optional[Dict[str, Any]] = None) -> BaseCaptionModel:
    """Create a model instance."""
    return ModelRegistry.create(name, config)


def list_models() -> List[str]:
    """List all registered models."""
    return ModelRegistry.list_models()

