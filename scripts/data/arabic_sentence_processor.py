import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import re

from camel_tools.utils.normalize import (
    normalize_unicode,
    normalize_alef_maksura_ar,
    normalize_alef_ar,
    normalize_teh_marbuta_ar
)
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer
from scripts.data.word import simple_word_detokenize

logger = logging.getLogger(__name__)

TATWEEL_MAP = CharMapper({
    u'\u0640': u''
})

class OrthographicFormat(Enum):
    """Orthographic format options"""
    ARABIC = "arabic"
    BUCKWALTER = "buckwalter"
    HSB = "hsb"  # Habash-Soudi-Buckwalter

class DiacriticFormat(Enum):
    """Diacritic handling options"""
    ORIGINAL = "original"  # Keep original diacritics
    DEDIACRITIZED = "dediacritized"  # Remove diacritics
    DIACRITIZED = "diacritized"  # Add diacritics using disambiguator

class MorphologicalScheme(Enum):
    """Morphological tokenization schemes"""
    WORD = "word"  # No morphological tokenization
    LEX = "lex"  # Replace with lemmas
    D3TOK = "d3tok"  # Morphological tokenization
    D3LEX = "d3lex"  # D3 tokenization with lemmas

class EncodingScheme(Enum):
    """Encoding schemes for morphological boundaries"""
    DEFAULT = "default"  # +_ markers
    SPACE = "space"  # Replace with spaces
    TATWEEL = "tatweel"  # Replace with tatweel (ـ)
    REMOVE = "remove"  # Remove markers completely

@dataclass
class ProcessingConfig:
    """Configuration for Arabic sentence processing"""
    # Unicode normalization
    unicode_normalize: bool = False
    
    # Orthographic normalization
    orthographic_normalize: bool = False
    orthographic_format: OrthographicFormat = OrthographicFormat.ARABIC
    
    # Diacritic handling
    diacritic_format: DiacriticFormat = DiacriticFormat.ORIGINAL
    
    # Morphological processing
    morphological_scheme: MorphologicalScheme = MorphologicalScheme.WORD
    morphological_split: bool = False  # Whether to split morphological tokens

    # Encoding scheme for morphological boundaries
    encoding_scheme: EncodingScheme = EncodingScheme.DEFAULT
    replacement_char: str = " "  # Character to use for replacement
    char_count: int = 1  # Number of replacement characters

class ArabicSentenceProcessor:
    """
    Comprehensive Arabic sentence processor with configurable transformations
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._setup_tools()
    
    def _setup_tools(self):
        """Initialize CAMeL Tools components based on configuration"""
        # Character mappers for transliteration
        if self.config.orthographic_format == OrthographicFormat.BUCKWALTER:
            self.ar2transliteration = CharMapper.builtin_mapper('ar2bw')
            self.transliteration2ar = CharMapper.builtin_mapper('bw2ar')
        elif self.config.orthographic_format == OrthographicFormat.HSB:
            self.ar2transliteration = CharMapper.builtin_mapper('ar2hsb')
            self.transliteration2ar = CharMapper.builtin_mapper('hsb2ar')
        
        # Morphological tools
        if (self.config.diacritic_format == DiacriticFormat.DIACRITIZED or 
            self.config.morphological_scheme != MorphologicalScheme.WORD):
            logger.info("Loading CAMeL Tools disambiguator...")
            self.mle = MLEDisambiguator.pretrained('calima-msa-r13')
            
            if self.config.morphological_scheme in [MorphologicalScheme.D3TOK, MorphologicalScheme.D3LEX]:
                logger.info("Setting up morphological tokenizer...")
                self.morphological_tokenizer = MorphologicalTokenizer(
                    self.mle,
                    scheme='d3tok',
                    split=self.config.morphological_split,
                    diac=self.config.diacritic_format == DiacriticFormat.DIACRITIZED,
                )
    
    def process(self, sentence: str) -> str:
        """
        Process a sentence according to the configuration
        
        Args:
            sentence: Input Arabic sentence
            
        Returns:
            Processed sentence
        """
        # Step 1: Unicode normalization
        if self.config.unicode_normalize:
            sentence = self._apply_unicode_normalization(sentence)
        
        # Step 2: Orthographic normalization
        if self.config.orthographic_normalize:
            sentence = self._apply_orthographic_normalization(sentence)
        
        # Step 3: Morphological processing (includes diacritic handling)
        sentence = self._apply_morphological_processing(sentence)
        
        # Step 4: Orthographic format conversion
        sentence = self._apply_orthographic_format(sentence)
        
        # Step 5: Encoding scheme application
        sentence = self._apply_encoding_scheme(sentence)
        
        return sentence
    
    def _apply_unicode_normalization(self, sentence: str) -> str:
        """Apply Unicode normalization"""
        return TATWEEL_MAP.map_string(normalize_unicode(sentence))

    
    def _apply_orthographic_normalization(self, sentence: str) -> str:
        """Apply orthographic normalization"""
        # Normalize alef variants
        sentence = normalize_alef_ar(sentence)
        # Normalize alef maksura to yeh
        sentence = normalize_alef_maksura_ar(sentence)
        # Normalize teh marbuta to heh
        sentence = normalize_teh_marbuta_ar(sentence)
        return sentence
    
    def _apply_morphological_processing(self, sentence: str) -> str:
        """Apply morphological processing and diacritic handling"""
        if self.config.morphological_scheme == MorphologicalScheme.WORD:
            # Handle diacritics only
            if self.config.diacritic_format == DiacriticFormat.DEDIACRITIZED:
                return dediac_ar(sentence)
            elif self.config.diacritic_format == DiacriticFormat.DIACRITIZED:
                return self._apply_diacritization(sentence)
            else:
                return sentence
        
        # Morphological processing
        word_tokens = simple_word_tokenize(sentence)
        
        if self.config.morphological_scheme == MorphologicalScheme.LEX:
            return self._apply_lemmatization(word_tokens)
        elif self.config.morphological_scheme in [MorphologicalScheme.D3TOK, MorphologicalScheme.D3LEX]:
            return self._apply_morphological_tokenization(word_tokens)
        
        return sentence
    
    def _apply_diacritization(self, sentence: str) -> str:
        """Apply diacritization using disambiguator"""
        word_tokens = simple_word_tokenize(sentence)
        disambig = self.mle.disambiguate(word_tokens)
        diacritized_words = [d.analyses[0].analysis['diac'] for d in disambig]
        return ' '.join(diacritized_words)
    
    def _apply_lemmatization(self, word_tokens: List[str]) -> str:
        """Replace words with their lemmas"""
        disambig = self.mle.disambiguate(word_tokens)
        lemmas = [d.analyses[0].analysis['lex'] for d in disambig]
        return ' '.join(lemmas)
    
    def _apply_morphological_tokenization(self, word_tokens: List[str]) -> str:
        """Apply morphological tokenization"""
        morph_tokens = self.morphological_tokenizer.tokenize(word_tokens)
        
        if self.config.morphological_scheme == MorphologicalScheme.D3LEX:
            # Replace base forms with lemmas
            morph_tokens = self._replace_base_with_lemmas(word_tokens, morph_tokens)
        
        # Handle diacritics
        if self.config.diacritic_format == DiacriticFormat.DEDIACRITIZED:
            morph_tokens = [dediac_ar(token) for token in morph_tokens]
        
        # Convert back to sentence if not split
        # if not self.config.morphological_split:
        return simple_word_detokenize(morph_tokens)
        # else:
            # return ' '.join(morph_tokens)
    
    def _replace_base_with_lemmas(self, word_tokens: List[str], morph_tokens: List[str]) -> List[str]:
        """Replace base forms in morphological tokens with lemmas (D3LEX scheme)"""
        disambig = self.mle.disambiguate(word_tokens)
        lemmas = [d.analyses[0].analysis['lex'] for d in disambig]
        
        # This is a simplified implementation
        # In practice, you'd need to identify which tokens are base forms
        result = []
        word_idx = 0
        
        for token in morph_tokens:
            if '+' not in token and '_' not in token:  # Likely a base form
                if word_idx < len(lemmas):
                    result.append(lemmas[word_idx])
                    word_idx += 1
                else:
                    result.append(token)
            else:
                result.append(token)
        
        return result
    
    def _apply_orthographic_format(self, sentence: str) -> str:
        """Apply orthographic format conversion"""
        if self.config.orthographic_format == OrthographicFormat.ARABIC:
            return sentence
        elif hasattr(self, 'ar2transliteration'):
            return self.ar2transliteration(sentence)
        return sentence
    
    def _apply_encoding_scheme(self, sentence: str) -> str:
        """Apply encoding scheme for morphological boundaries"""
        if self.config.encoding_scheme == EncodingScheme.DEFAULT:
            return sentence
        
        replacement = ""
        if self.config.encoding_scheme == EncodingScheme.SPACE:
            replacement = " " * self.config.char_count
        elif self.config.encoding_scheme == EncodingScheme.TATWEEL:
            replacement = "ـ" * self.config.char_count  # Arabic tatweel
        elif self.config.encoding_scheme == EncodingScheme.REMOVE:
            replacement = ""
        else:
            replacement = self.config.replacement_char * self.config.char_count
        
        # Replace both +_ and _+ patterns
        sentence = sentence.replace('+_', replacement).replace('_+', replacement)
        return sentence
    
    def get_analysis_info(self, sentence: str) -> Dict[str, Any]:
        """
        Get detailed analysis information for debugging/inspection
        
        Returns:
            Dictionary with analysis details
        """
        word_tokens = simple_word_tokenize(sentence)
        info = {
            'original_sentence': sentence,
            'word_tokens': word_tokens,
            'processing_steps': []
        }
        
        # Add analysis if morphological tools are available
        if hasattr(self, 'mle'):
            disambig = self.mle.disambiguate(word_tokens)
            info['morphological_analysis'] = []
            
            for word, disambiguation in zip(word_tokens, disambig):
                analysis = disambiguation.analyses[0].analysis
                info['morphological_analysis'].append({
                    'word': word,
                    'diacritized': analysis.get('diac', ''),
                    'lemma': analysis.get('lex', ''),
                    'pos': analysis.get('pos', ''),
                    'features': analysis.get('feat', ''),
                    'gloss': analysis.get('gloss', ''),
                    'root': analysis.get('root', '')
                })
        
        return info

# Convenience functions for common configurations
def create_default_config() -> ProcessingConfig:
    """Create default processing configuration"""
    return ProcessingConfig()

def create_normalized_config() -> ProcessingConfig:
    """Create configuration for normalized text"""
    return ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=True,
        diacritic_format=DiacriticFormat.DEDIACRITIZED
    )

def create_diacritized_config() -> ProcessingConfig:
    """Create configuration for diacritized text"""
    return ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=True,
        diacritic_format=DiacriticFormat.DIACRITIZED
    )

def create_morphological_config(scheme: MorphologicalScheme = MorphologicalScheme.D3TOK) -> ProcessingConfig:
    """Create configuration for morphological processing"""
    return ProcessingConfig(
        unicode_normalize=True,
        orthographic_normalize=True,
        morphological_scheme=scheme,
        morphological_split=False,
        diacritic_format=DiacriticFormat.DEDIACRITIZED
    )

def create_buckwalter_config() -> ProcessingConfig:
    """Create configuration for Buckwalter transliteration"""
    return ProcessingConfig(
        unicode_normalize=True,
        orthographic_format=OrthographicFormat.BUCKWALTER,
        diacritic_format=DiacriticFormat.DEDIACRITIZED
    )

# Example usage and testing
def test_processor():
    """Test the processor with various configurations"""
    test_sentence = "هَـــلْ ذَهَبْتَ إِلَى المَكْتَبَةِ؟"
    print("Original sentence:", test_sentence)
    print("=" * 60)
    
    # Test different configurations
    configs = [
        ("Default", create_default_config()),
        ("Normalized", create_normalized_config()),
        ("Diacritized", create_diacritized_config()),
        ("D3TOK Morphological", create_morphological_config(MorphologicalScheme.D3TOK)),
        ("LEX Morphological", create_morphological_config(MorphologicalScheme.D3LEX)),
        ("Buckwalter", create_buckwalter_config()),
    ]
    
    for name, config in configs:
        processor = ArabicSentenceProcessor(config)
        result = processor.process(test_sentence)
        print(f"{name}: {result}")
    
    # Test encoding schemes with morphological tokenization
    print("\n" + "=" * 60)
    print("Testing encoding schemes:")
    
    base_config = create_morphological_config(MorphologicalScheme.D3TOK)
    encoding_schemes = [
        ("Default (+_)", EncodingScheme.DEFAULT),
        ("Space", EncodingScheme.SPACE),
        ("Tatweel", EncodingScheme.TATWEEL),
        ("Remove", EncodingScheme.REMOVE),
    ]
    
    for name, scheme in encoding_schemes:
        config = ProcessingConfig(
            unicode_normalize=True,
            orthographic_normalize=True,
            morphological_scheme=MorphologicalScheme.D3TOK,
            morphological_split=False,
            diacritic_format=DiacriticFormat.DEDIACRITIZED,
            encoding_scheme=scheme
        )
        processor = ArabicSentenceProcessor(config)
        result = processor.process(test_sentence)
        print(f"{name}: {result}")

if __name__ == "__main__":
    test_processor()