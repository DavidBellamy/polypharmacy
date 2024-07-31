TEST_MODE = True 

if TEST_MODE:
    PPI_PATH = "data/test/bio-decagon-ppi.csv"
    COMBO_SIDE_EFFECT_PATH = "data/test/bio-decagon-combo.csv"
    DRUG_GENE_PATH = "data/test/bio-decagon-targets.csv"
    MONO_SIDE_EFFECT_PATH = "data/test/bio-decagon-mono.csv"
else:
    PPI_PATH = "data/bio-decagon-ppi.csv"
    COMBO_SIDE_EFFECT_PATH = "data/bio-decagon-combo.csv"
    DRUG_GENE_PATH = "data/bio-decagon-targets.csv"
    MONO_SIDE_EFFECT_PATH = "data/bio-decagon-mono.csv"

__all__ = ['TEST_MODE', 'PPI_PATH', 'COMBO_SIDE_EFFECT_PATH', 'DRUG_GENE_PATH', 'MONO_SIDE_EFFECT_PATH']
