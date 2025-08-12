from ..processing.arabic_sentence_processor import (
    ProcessingConfig,
    OrthographicFormat,
    DiacriticFormat,
    MorphologicalScheme,
    EncodingScheme,
)

# Base configuration with common settings
BASE_CONFIG = {
    "unicode_normalize": True,  # Always apply unicode normalization
}

# Complete Orthographic Experiment Configurations (3×2×2 = 12 configs)
ORTHOGRAPHIC_CONFIGS = {
    # Arabic Script Configurations (4 configs)
    "arabic_default": ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=False,
        orthographic_format=OrthographicFormat.ARABIC,
        diacritic_format=DiacriticFormat.ORIGINAL
    ),
    "arabic_norm_dediac": ProcessingConfig( # This is the common practice
        unicode_normalize=True,
        orthographic_normalize=True,
        orthographic_format=OrthographicFormat.ARABIC,
        diacritic_format=DiacriticFormat.DEDIACRITIZED
    ),
    "arabic_nonorm_diac": ProcessingConfig( # This has maximal disambiguity
        unicode_normalize=True,
        orthographic_normalize=False,
        orthographic_format=OrthographicFormat.ARABIC,
        diacritic_format=DiacriticFormat.DIACRITIZED
    ),

    # Buckwalter Script Configurations (3 configs)
    "buckwalter_default": ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=False,
        orthographic_format=OrthographicFormat.BUCKWALTER,
        diacritic_format=DiacriticFormat.ORIGINAL
    ),
    "buckwalter_norm_dediac": ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=True,
        orthographic_format=OrthographicFormat.BUCKWALTER,
        diacritic_format=DiacriticFormat.DEDIACRITIZED
    ),
    "buckwalter_nonorm_diac": ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=False,
        orthographic_format=OrthographicFormat.BUCKWALTER,
        diacritic_format=DiacriticFormat.DIACRITIZED
    ),

    # HSB Script Configurations (4 configs)
    "hsb_default": ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=False,
        orthographic_format=OrthographicFormat.HSB,
        diacritic_format=DiacriticFormat.ORIGINAL
    ),
    "hsb_nonorm_diac": ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=False,
        orthographic_format=OrthographicFormat.HSB,
        diacritic_format=DiacriticFormat.DIACRITIZED
    ),
    "hsb_norm_dediac": ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=True,
        orthographic_format=OrthographicFormat.HSB,
        diacritic_format=DiacriticFormat.DEDIACRITIZED
    )
}

best_orthographic_config = {
    "unicode_normalize": True,
    "orthographic_normalize": False,
    "orthographic_format": OrthographicFormat.ARABIC,
    "diacritic_format": DiacriticFormat.ORIGINAL
}

# Morphological Experiment Configurations (4×4 = 16 configs)
MORPHOLOGICAL_CONFIGS = {
    # Word-level (baseline) configurations
    "morph_word": ProcessingConfig(
        **best_orthographic_config,
        morphological_scheme=MorphologicalScheme.WORD
    ),
    "morph_lex": ProcessingConfig(
        **best_orthographic_config,
        morphological_scheme=MorphologicalScheme.LEX
    ),
    
    # Lemmatization (LEX) configurations
    "morph_d3tok_default": ProcessingConfig(
        **best_orthographic_config,
        morphological_scheme=MorphologicalScheme.D3TOK,
        encoding_scheme=EncodingScheme.DEFAULT
    ),
    "morph_d3tok_tatweel": ProcessingConfig(
        **best_orthographic_config,
        morphological_scheme=MorphologicalScheme.D3TOK,
        encoding_scheme=EncodingScheme.TATWEEL
    ),
    
    # D3TOK configurations with different encoding schemes
    "morph_d3tok_space": ProcessingConfig(
        **best_orthographic_config,
        morphological_scheme=MorphologicalScheme.D3TOK,
        encoding_scheme=EncodingScheme.SPACE
    ),
}

# Combine all configurations
ALL_CONFIGS = {
    **ORTHOGRAPHIC_CONFIGS,
    **MORPHOLOGICAL_CONFIGS,
}

def get_processing_config(config_name: str) -> ProcessingConfig:
    """
    Get a processing configuration by name.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        ProcessingConfig object
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in ALL_CONFIGS:
        available_configs = list(ALL_CONFIGS.keys())
        raise ValueError(
            f"Configuration '{config_name}' not found. "
            f"Available configurations: {available_configs}"
        )
    
    return ALL_CONFIGS[config_name]

