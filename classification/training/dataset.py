from classification.datasets import TCGA_NSCLC, tcga_nsclc_collate, TCGA_RCC, tcga_rcc_collate


def get_dataset(cfg, fold=0):
    if cfg.dataset.name == 'TCGA-NSCLC':
        train_dataset = TCGA_NSCLC(root_path=cfg.dataset.root_path, split='train', cfg=cfg, fold=fold, num_samples=cfg.meta.num_samples)
        validation_dataset = TCGA_NSCLC(root_path=cfg.dataset.root_path, split='val', cfg=cfg, fold=fold, num_samples=cfg.meta.num_samples)
        test_dataset = TCGA_NSCLC(root_path=cfg.dataset.root_path, split='test', cfg=cfg, fold=fold, num_samples=cfg.meta.num_samples)
        collate_fn = tcga_nsclc_collate

    if cfg.dataset.name == 'TCGA-RCC':
        train_dataset = TCGA_RCC(root_path=cfg.dataset.root_path, split='train', cfg=cfg, fold=fold, num_samples=cfg.meta.num_samples)
        validation_dataset = TCGA_RCC(root_path=cfg.dataset.root_path, split='val', cfg=cfg, fold=fold, num_samples=cfg.meta.num_samples)
        test_dataset = TCGA_RCC(root_path=cfg.dataset.root_path, split='test', cfg=cfg, fold=fold, num_samples=cfg.meta.num_samples)
        collate_fn = tcga_rcc_collate

    return train_dataset, validation_dataset, test_dataset, collate_fn
