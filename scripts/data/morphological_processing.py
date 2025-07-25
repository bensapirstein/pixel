import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer
import re
from typing import List, Dict, Any
from word import simple_word_detokenize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_camel_tools():
    """Initialize CAMeL Tools morphological analyzer and tokenizer"""
    logger.info("Loading CAMeL Tools morphological database...")
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)
    
    logger.info("Loading CAMeL Tools disambiguator...")
    mle = MLEDisambiguator.pretrained('calima-msa-r13')
    
    logger.info("Setting up morphological tokenizer...")
    # Create tokenizers for different variants
    tokenizer_undiac = MorphologicalTokenizer(mle, scheme='d3tok', split=False, diac=False)
    tokenizer_diac = MorphologicalTokenizer(mle, scheme='d3tok', split=False, diac=True)
    
    logger.info("CAMeL Tools setup completed successfully")
    return analyzer, tokenizer_undiac, tokenizer_diac

def analyze_word(word: str, analyzer: Analyzer) -> List[Dict[str, Any]]:
    """Analyze a single Arabic word using CAMeL Tools"""
    # Clean the word (remove punctuation for analysis)
    clean_word = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', '', word)
    
    if not clean_word:
        return []
    
    try:
        analyses = analyzer.analyze(clean_word)
        return [
            {
                'word': word,
                'clean_word': clean_word,
                'analysis': str(analysis),
                'lemma': analysis.get('lex', ''),
                'pos': analysis.get('pos', ''),
                'features': analysis.get('feat', ''),
                'gloss': analysis.get('gloss', ''),
                'root': analysis.get('root', ''),
                'pattern': analysis.get('pattern', ''),
                'caphi': analysis.get('caphi', ''),
                'catib6': analysis.get('catib6', '')
            }
            for analysis in analyses
        ]
    except Exception as e:
        logger.warning(f"Error analyzing word '{word}': {e}")
        return []

def process_sentence(sentence: str, analyzer: Analyzer, tokenizer_undiac: MorphologicalTokenizer, tokenizer_diac: MorphologicalTokenizer) -> Dict[str, Any]:
    """Process a complete sentence with morphological analysis and tokenization"""
    # Simple word tokenization first
    word_tokens = simple_word_tokenize(sentence)
    
    # Morphological tokenization
    try:
        d3tok_undiacritized = tokenizer_undiac.tokenize(word_tokens)
        d3tok_diacritized = tokenizer_diac.tokenize(word_tokens)
    except Exception as e:
        logger.warning(f"Error in morphological tokenization for sentence '{sentence[:50]}...': {e}")
        d3tok_undiacritized = word_tokens  # fallback to word tokens
        d3tok_diacritized = word_tokens
    
    sentence_analysis = {
        'original_sentence': sentence,
        'word_tokens': word_tokens,
        'd3tok_undiacritized': d3tok_undiacritized,
        'd3tok_diacritized': d3tok_diacritized,
        'morphological_analyses': []
    }
    
    # Perform morphological analysis on word tokens
    for token in word_tokens:
        token_analyses = analyze_word(token, analyzer)
        sentence_analysis['morphological_analyses'].append(token_analyses)
    
    return sentence_analysis

def process_barec_dataset(dataset_name: str, debug_num_examples: int = None) -> DatasetDict:
    """Process the BAREC dataset with morphological analysis and tokenization"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # If debug mode, limit the number of examples
    if debug_num_examples is not None:
        logger.info(f"DEBUG MODE: Processing only {debug_num_examples} examples per split")
        for split_name in dataset.keys():
            dataset[split_name] = dataset[split_name].select(range(min(debug_num_examples, len(dataset[split_name]))))
    
    # Initialize CAMeL Tools
    analyzer, tokenizer_undiac, tokenizer_diac = setup_camel_tools()
    
    def process_batch(batch):
        """Process a batch of examples"""
        processed_sentences = []
        
        for sentence in batch['Sentence']:
            try:
                analysis = process_sentence(sentence, analyzer, tokenizer_undiac, tokenizer_diac)
                processed_sentences.append(analysis)
            except Exception as e:
                logger.error(f"Error processing sentence '{sentence[:50]}...': {e}")
                # Add empty analysis for failed sentences
                processed_sentences.append({
                    'original_sentence': sentence,
                    'word_tokens': [],
                    'd3tok_undiacritized': [],
                    'd3tok_diacritized': [],
                    'morphological_analyses': []
                })
        
        return {
            'Sentence': batch['Sentence'],
            'Readability_Level_19': batch['Readability_Level_19'],
            'morphological_analysis': processed_sentences
        }
    
    # Process each split
    processed_dataset = {}
    for split_name, split_data in dataset.items():
        logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
        
        # Process in batches to show progress
        processed_split = split_data.map(
            process_batch,
            batched=True,
            batch_size=5,  # Reduced batch size for stability with morphological tokenization
            desc=f"Processing {split_name}"
        )
        
        processed_dataset[split_name] = processed_split
    
    return DatasetDict(processed_dataset)

def create_final_dataset(processed_dataset: DatasetDict) -> DatasetDict:
    """Create the final dataset with simplified morphological features and d3tok tokenization"""
    
    def create_final_batch(batch):
        final_analyses = []
        
        for analysis in batch['morphological_analysis']:
            # Extract simplified morphological features
            simplified_morph = {
                'lemmas': [],
                'pos_tags': [],
                'roots': [],
                'features': []
            }
            
            for token_analyses in analysis['morphological_analyses']:
                if token_analyses:
                    # Take the first analysis (most likely)
                    first_analysis = token_analyses[0]
                    simplified_morph['lemmas'].append(first_analysis['lemma'])
                    simplified_morph['pos_tags'].append(first_analysis['pos'])
                    simplified_morph['roots'].append(first_analysis['root'])
                    simplified_morph['features'].append(first_analysis['features'])
                else:
                    # No analysis found
                    simplified_morph['lemmas'].append('')
                    simplified_morph['pos_tags'].append('')
                    simplified_morph['roots'].append('')
                    simplified_morph['features'].append('')
            
            # Combine all analysis
            final_analysis = {
                'original_sentence': analysis['original_sentence'],
                'word_tokens': analysis['word_tokens'],
                'd3tok_undiacritized': analysis['d3tok_undiacritized'],
                'd3tok_diacritized': analysis['d3tok_diacritized'],
                'lemmas': simplified_morph['lemmas'],
                'pos_tags': simplified_morph['pos_tags'],
                'roots': simplified_morph['roots'],
                'features': simplified_morph['features']
            }
            
            final_analyses.append(final_analysis)
        
        return {
            'Sentence': batch['Sentence'],
            'Readability_Level_19': batch['Readability_Level_19'],
            'morphological_analysis': final_analyses
        }
    
    final_dataset = {}
    for split_name, split_data in processed_dataset.items():
        logger.info(f"Creating final version of {split_name} split...")
        final_dataset[split_name] = split_data.map(
            create_final_batch,
            batched=True,
            desc=f"Finalizing {split_name}"
        )
    
    return DatasetDict(final_dataset)

def upload_to_hub(
    dataset: DatasetDict, 
    repo_name: str, 
    hf_token: str = None,
    private: bool = False
):
    """Upload the processed dataset to Hugging Face Hub"""
    
    if hf_token is None:
        # Try to get token from environment
        hf_token = os.getenv('HF_TOKEN')
        if hf_token is None:
            raise ValueError("HF_TOKEN not found in environment or provided as parameter")
    
    logger.info(f"Uploading dataset to: {repo_name}")
    
    try:
        dataset.push_to_hub(
            repo_name,
            token=hf_token,
            private=private
        )
        logger.info(f"Successfully uploaded dataset to {repo_name}")
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise

def print_debug_sample(dataset: DatasetDict, num_examples: int = 10):
    """Print sample processed examples for debugging"""
    logger.info("DEBUG: Sample processed examples:")
    
    for split_name, split_data in dataset.items():
        logger.info(f"\n--- {split_name.upper()} SPLIT ---")
        for i in range(min(num_examples, len(split_data))):
            example = split_data[i]
            morph_analysis = example['morphological_analysis']
            
            print(f"\nExample {i+1}:")
            print(f"Original sentence: {morph_analysis['original_sentence']}")
            print(f"Word tokens: {morph_analysis['word_tokens']}")
            print(f"D3tok undiacritized: {morph_analysis['d3tok_undiacritized']}")
            print(f"D3tok diacritized: {morph_analysis['d3tok_diacritized']}")
            print(f"Lemmas: {morph_analysis['lemmas']}")
            print(f"POS tags: {morph_analysis['pos_tags']}")
            print(f"Roots: {morph_analysis['roots']}")
            print(f"Label: {example['Readability_Level_19']}")
            print("-" * 50)

def main():
    """Main processing pipeline"""
    parser = argparse.ArgumentParser(description="Process BAREC dataset with morphological analysis")
    parser.add_argument(
        "--debug", 
        type=int, 
        default=None,
        help="Number of examples to process for debugging (skips upload)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="CAMeL-Lab/BAREC-Shared-Task-2025-sent",
        help="Name of the dataset to process"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="bensapir/BAREC-Shared-Task-2025-sent-morphological",
        help="Repository name for uploading to Hub"
    )
    
    args = parser.parse_args()
    
    # Process the BAREC dataset
    processed_dataset = process_barec_dataset(args.dataset_name, debug_num_examples=args.debug)
    
    # Create final version with d3tok tokenization and simplified morphology
    final_dataset = create_final_dataset(processed_dataset)
    
    # If in debug mode, print sample and skip upload
    if args.debug is not None:
        logger.info(f"DEBUG MODE: Processed {args.debug} examples per split")
        print_debug_sample(final_dataset)
        logger.info("DEBUG MODE: Skipping upload to Hub")
        return

    # Upload the dataset (only in non-debug mode)
    hf_token = os.getenv('HF_TOKEN')
    
    upload_to_hub(
        final_dataset,
        args.repo_name,
        hf_token=hf_token,
        private=False
    )
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()