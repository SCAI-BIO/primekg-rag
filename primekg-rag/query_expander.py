import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Set
import logging

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MedicalQueryExpander:
    def __init__(self):
        # Medical term mappings (can be extended)
        self.medical_synonyms = {
            'heart attack': ['myocardial infarction', 'cardiac arrest', 'MI', 'acute coronary syndrome'],
            'cancer': ['malignancy', 'neoplasm', 'carcinoma', 'tumor'],
            'diabetes': ['diabetes mellitus', 'high blood sugar', 'hyperglycemia'],
            'bp': ['blood pressure', 'hypertension', 'high blood pressure'],
        }
        
    def expand_term(self, term: str) -> Set[str]:
        """
        Expand a single term using WordNet and medical synonyms.
        Returns a set of related terms including the original.
        """
        expanded_terms = {term.lower().strip()}
        
        # Check medical synonyms first
        for key, values in self.medical_synonyms.items():
            if term.lower() == key.lower():
                expanded_terms.update(values)
            elif term.lower() in [v.lower() for v in values]:
                expanded_terms.update(values)
                expanded_terms.add(key)
        
        # Use WordNet for general synonyms
        for syn in wn.synsets(term):
            # Get lemmas and add to terms
            for lemma in syn.lemmas():
                expanded_terms.add(lemma.name().replace('_', ' '))
            
            # Add hypernyms (more general terms)
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    expanded_terms.add(lemma.name().replace('_', ' '))
            
            # Add hyponyms (more specific terms)
            for hypo in syn.hyponyms():
                for lemma in hypo.lemmas():
                    expanded_terms.add(lemma.name().replace('_', ' '))
        
        return {t for t in expanded_terms if t and len(t) > 1}
    
    def expand_query(self, query: str) -> str:
        """
        Expand a search query by finding synonyms and related terms.
        Returns an expanded query string.
        """
        if not query or not query.strip():
            return query
            
        terms = query.split()
        expanded_terms = []
        
        # Expand each term individually
        for term in terms:
            expanded = self.expand_term(term)
            if expanded:
                # Group synonyms with OR
                if len(expanded) > 1:
                    expanded_terms.append(f"({' OR '.join(expanded)})")
                else:
                    expanded_terms.append(expanded.pop())
        
        # Combine with AND between original terms
        expanded_query = ' AND '.join(expanded_terms)
        
        logger.debug(f"Expanded query: {query} -> {expanded_query}")
        return expanded_query

# Example usage
if __name__ == "__main__":
    expander = MedicalQueryExpander()
    
    test_queries = [
        "heart attack treatment",
        "diabetes symptoms",
        "cancer research"
    ]
    
    for query in test_queries:
        expanded = expander.expand_query(query)
        print(f"Original: {query}")
        print(f"Expanded: {expanded}\n")
