from classification.models import CATExplainable


def get_model(cfg):
    if cfg.model.name == 'CATExplainable':
        return CATExplainable(
                num_classes=cfg.model.output_logits, 
                **cfg.model.meta,
                )
        
    raise KeyError(f'Model {cfg.model.name} is not supported')
