import re
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
from camel_tools.utils.charsets import EMOJI_MULTICHAR_CHARSET
from camel_tools.tokenizers.word import simple_word_tokenize

def simple_word_detokenize(tokens):
    """Reconstructs a sentence from tokens by adding appropriate spacing."""
    if not tokens:
        return ""
    
    _PUNCT_SYMBOLS = UNICODE_PUNCT_SYMBOL_CHARSET | EMOJI_MULTICHAR_CHARSET
    opening_punct = {'(', '[', '{', '«'}
    closing_punct = {')', ']', '}', '»', '.', '!', '?', '،', '؛', '؟'}
    quotes = {'"', "'"}  # Handle quotes separately
    # Punctuation that should ALWAYS have space after them
    always_space_after = {':'}
    # Punctuation that should have space after them only if followed by words/opening punct
    conditional_space_after = {',', '،', ';', '؛'}
    
    def is_punctuation(token):
        return token in _PUNCT_SYMBOLS or len(token) == 1 and any(char in _PUNCT_SYMBOLS for char in token)
    
    def is_word(token):
        return not is_punctuation(token)
    
    def ends_with_plus(token):
        return token.endswith('+')
    
    def starts_with_plus(token):
        return token.startswith('+')
    
    result = []
    for i, token in enumerate(tokens):
        result.append(token)
        
        # Skip space after last token
        if i == len(tokens) - 1:
            continue
            
        next_token = tokens[i + 1]
        
        # Special handling for + morpheme boundaries
        # No space after words ending with +
        if ends_with_plus(token):
            continue
        
        # No space before words starting with +
        if starts_with_plus(next_token):
            continue
        
        # Rules for adding spaces:
        
        # 1. ALWAYS space after colon
        if token in always_space_after:
            result.append(' ')
        
        # 2. Space after closing punct if next is word or opening punct (but not quotes)
        elif (token in closing_punct and 
            (is_word(next_token) or next_token in opening_punct)):
            result.append(' ')
            
        # 3. Space after comma/semicolon ONLY if next is word or opening punct
        elif (token in conditional_space_after and 
              (is_word(next_token) or next_token in opening_punct)):
            result.append(' ')
            
        # 4. Space before opening punct if current is word
        elif (is_word(token) and next_token in opening_punct):
            result.append(' ')
            
        # 5. Space between words
        elif (is_word(token) and is_word(next_token)):
            result.append(' ')
            
        # 6. No space around quotes or between punctuation
    
    return ''.join(result)

def smart_word_detokenize(tokens):
    """Enhanced detokenization with better Arabic text handling."""
    if not tokens:
        return ""
    
    opening_punct = {'(', '[', '{', '«'}
    closing_punct = {')', ']', '}', '»', '.', '!', '?', '،', '؛', '؟'}
    quotes = {'"', "'"}  # Handle quotes separately
    # Punctuation that should ALWAYS have space after them
    always_space_after = {':'}
    # Punctuation that should have space after them only if followed by words/opening punct
    conditional_space_after = {',', '،', ';', '؛'}
    all_punct = opening_punct | closing_punct | always_space_after | conditional_space_after | quotes
    
    def is_word(token):
        return token not in all_punct
    
    def ends_with_plus(token):
        return token.endswith('+')
    
    def starts_with_plus(token):
        return token.startswith('+')
    
    result = []
    for i, token in enumerate(tokens):
        result.append(token)
        
        # Skip space after last token
        if i == len(tokens) - 1:
            continue
            
        next_token = tokens[i + 1]
        
        # Special handling for + morpheme boundaries
        # No space after words ending with +
        if ends_with_plus(token):
            continue
        
        # No space before words starting with +
        if starts_with_plus(next_token):
            continue
        
        # Rules for adding spaces:
        
        # 1. ALWAYS space after colon
        if token in always_space_after:
            result.append(' ')
        
        # 2. Space after closing punct if next is word or opening punct (but not quotes)
        elif (token in closing_punct and 
            (is_word(next_token) or next_token in opening_punct)):
            result.append(' ')
            
        # 3. Space after comma/semicolon ONLY if next is word or opening punct
        elif (token in conditional_space_after and 
              (is_word(next_token) or next_token in opening_punct)):
            result.append(' ')
            
        # 4. Space before opening punct if current is word
        elif (is_word(token) and next_token in opening_punct):
            result.append(' ')
            
        # 5. Space between words
        elif (is_word(token) and is_word(next_token)):
            result.append(' ')
            
        # 6. No space around quotes or between punctuation
    
    return ''.join(result)

def test_detokenization():
    """Test both detokenization methods with roundtrip tokenization"""
    
    test_sentences = [
        "Hello, world!!!",
        "هذا نص عربي، وهو جميل.",
        "(مرحبا) بك في العالم",
        "السلام عليكم؟ كيف حالك!",
        'قال: "مرحبا".',
        "العدد 123 مهم جداً",
        "This is a mix: العربية والإنجليزية together!",
        "Test punctuation: ()[]{}\"'«»",
        "Arabic punctuation: ،؛؟!",
        "Simple sentence without punctuation",
        "Numbers 123 and 456 in text",
        "Multiple!!! exclamation!!! marks!!!",
        # Test cases for + morpheme boundaries
        "word+ +suffix morpheme",
        "prefix+ +middle+ +suffix",
        "normal word+ +attached",
    ]
    
    simple_success = smart_success = 0
    total = len(test_sentences)
    
    print("Testing detokenization methods:")
    print("=" * 80)
    
    for i, original in enumerate(test_sentences, 1):
        tokens = simple_word_tokenize(original)
        simple_result = simple_word_detokenize(tokens)
        smart_result = smart_word_detokenize(tokens)
        
        simple_match = original == simple_result
        smart_match = original == smart_result
        
        if simple_match: simple_success += 1
        if smart_match: smart_success += 1
        
        print(f"Test {i}: Simple {'✓' if simple_match else '✗'} | Smart {'✓' if smart_match else '✗'}")
        print(f"Original: '{original}'")
        if not simple_match or not smart_match:
            print(f"Tokens:   {tokens}")
            if not simple_match: print(f"Simple:   '{simple_result}'")
            if not smart_match: print(f"Smart:    '{smart_result}'")
        print("-" * 80)
    
    print(f"RESULTS:")
    print(f"Simple: {simple_success}/{total} ({simple_success/total*100:.1f}%)")
    print(f"Smart:  {smart_success}/{total} ({smart_success/total*100:.1f}%)")

if __name__ == "__main__":
    test_detokenization()