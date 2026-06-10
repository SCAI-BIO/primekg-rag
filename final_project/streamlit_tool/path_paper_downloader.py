"""
Path Paper Downloader Module

This module handles downloading PubMed papers for shortest path nodes,
extracting MeSH terms from PMC, and embedding them into a dedicated ChromaDB collection.
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from Bio import Entrez
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Configure Entrez
Entrez.email = "anamekawy@gmail.com"
Entrez.tool = "PathPaperDownloader"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SHORTEST_PATH_DB_PATH = "./shortest_path_db"
COLLECTION_NAME = "shortest_path_papers"
DEFAULT_PAPERS_PER_NODE = 10
DEFAULT_MAX_TOTAL_PAPERS = 50
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

class PathPaperDownloader:
    """Downloads and processes papers for shortest path nodes with PMC and MeSH terms."""
    
    def __init__(self, db_path: str = SHORTEST_PATH_DB_PATH):
        """Initialize the downloader with ChromaDB and embedding model."""
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = None
        self.collection = None
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            
            # Create embedding function with correct interface
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
            
            # Create or get collection with embedding function
            try:
                self.collection = self.chroma_client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embedding_function
                )
                logger.info(f"Using existing collection: {COLLECTION_NAME}")
            except Exception:
                # Create new collection
                self.collection = self.chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embedding_function
                )
                logger.info(f"Created new collection: {COLLECTION_NAME}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def fetch_pmids_for_term(self, term: str, retmax: int = 100) -> List[str]:
        """Fetch PMIDs for a given search term from PubMed."""
        logger.info(f"Fetching PMIDs for term: '{term}' (max: {retmax})")
        try:
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed", 
                term=term, 
                retmax=retmax, 
                retmode="xml"
            )
            results = Entrez.read(handle)
            handle.close()
            
            pmids = results['IdList']
            logger.info(f"Found {len(pmids)} PMIDs for term: '{term}'")
            
            # Log first few PMIDs for debugging
            if pmids:
                logger.info(f"Sample PMIDs: {pmids[:3]}")
            else:
                logger.warning(f"No PMIDs found for term: '{term}'")
            
            return pmids
            
        except Exception as e:
            logger.error(f"Error fetching PMIDs for term '{term}': {e}")
            return []
    
    def fetch_paper_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed paper information including MeSH terms from PubMed."""
        papers = []
        
        for i in tqdm(range(0, len(pmids), 100), desc="Fetching paper details"):
            batch = pmids[i:i+100]
            try:
                # Fetch paper details
                handle = Entrez.efetch(
                    db="pubmed", 
                    id=batch, 
                    retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()
                
                for article in records.get("PubmedArticle", []):
                    try:
                        paper_data = self._extract_paper_data(article)
                        if paper_data:
                            papers.append(paper_data)
                    except Exception as e:
                        logger.warning(f"Skipping article due to parse error: {e}")
                        
            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
            
            time.sleep(0.5)  # Rate limiting
        
        return papers
    
    def _extract_paper_data(self, article: Dict) -> Optional[Dict[str, Any]]:
        """Extract paper data from PubMed article record."""
        try:
            citation = article['MedlineCitation']
            pmid = str(citation['PMID'])
            
            # Extract basic info
            article_data = citation['Article']
            title = str(article_data.get('ArticleTitle', ''))
            
            # Extract abstract
            abstract_texts = article_data.get('Abstract', {}).get('AbstractText', [''])
            if isinstance(abstract_texts, list):
                abstract = ' '.join(str(part) for part in abstract_texts)
            else:
                abstract = str(abstract_texts)
            
            # Extract MeSH terms
            mesh_terms = []
            if 'MeshHeadingList' in citation:
                for mesh_heading in citation['MeshHeadingList']:
                    descriptor = mesh_heading.get('DescriptorName', '')
                    if descriptor:
                        mesh_terms.append(str(descriptor))
            
            # Extract journal info
            journal = article_data.get('Journal', {}).get('Title', '')
            
            # Extract publication date
            pub_date = self._extract_publication_date(article_data.get('Journal', {}))
            
            # Extract authors
            authors = self._extract_authors(article_data.get('AuthorList', []))
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'mesh_terms': mesh_terms,
                'journal': journal,
                'publication_date': pub_date,
                'authors': authors,
                'source': 'pubmed'
            }
            
        except Exception as e:
            logger.warning(f"Error extracting paper data: {e}")
            return None
    
    def _extract_publication_date(self, journal_data: Dict) -> str:
        """Extract publication date from journal data."""
        try:
            if 'JournalIssue' in journal_data:
                issue = journal_data['JournalIssue']
                if 'PubDate' in issue:
                    pub_date = issue['PubDate']
                    year = pub_date.get('Year', '')
                    month = pub_date.get('Month', '')
                    day = pub_date.get('Day', '')
                    
                    if year:
                        date_parts = [year]
                        if month:
                            date_parts.append(month)
                        if day:
                            date_parts.append(day)
                        return '-'.join(date_parts)
        except Exception:
            pass
        return ''
    
    def _extract_authors(self, author_list: List[Dict]) -> List[str]:
        """Extract author names from author list."""
        authors = []
        try:
            for author in author_list:
                if 'LastName' in author and 'ForeName' in author:
                    name = f"{author['ForeName']} {author['LastName']}"
                    authors.append(name)
                elif 'LastName' in author:
                    authors.append(author['LastName'])
        except Exception:
            pass
        return authors
    
    def fetch_pmc_metadata(self, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch PMC metadata (PMC ID and additional MeSH terms) for given PMIDs."""
        pmc_data = {}
        
        for pmid in tqdm(pmids, desc="Fetching PMC metadata"):
            try:
                # Get PMC ID from PMID
                pmc_id = self._get_pmc_id_from_pmid(pmid)
                if pmc_id:
                    pmc_data[pmid] = {
                        'pmc_id': pmc_id,
                        'pmc_url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
                    }
                    
            except Exception as e:
                logger.warning(f"Error fetching PMC metadata for PMID {pmid}: {e}")
            
            time.sleep(0.3)  # Rate limiting
        
        return pmc_data
    
    def _get_pmc_id_from_pmid(self, pmid: str) -> Optional[str]:
        """Get PMC ID from PMID using E-utilities."""
        try:
            # Use elink to find PMC ID
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=pmid,
                retmode="xml"
            )
            results = Entrez.read(handle)
            handle.close()
            
            # Extract PMC ID from results
            for link_set in results:
                if 'LinkSetDb' in link_set:
                    for link_db in link_set['LinkSetDb']:
                        if link_db['DbTo'] == 'pmc' and 'Link' in link_db:
                            for link in link_db['Link']:
                                pmc_id = link.get('Id')
                                if pmc_id:
                                    return pmc_id
        except Exception as e:
            logger.warning(f"Error getting PMC ID for PMID {pmid}: {e}")
        
        return None
    
    # Removed _fetch_pmc_article method since we only need abstracts
    
    def create_association_search_terms(self, source_node: str, target_node: str) -> List[str]:
        """Create search terms that focus on associations between source and target nodes."""
        # Start with simpler, more likely to succeed searches
        association_terms = [
            f'"{source_node}" AND "{target_node}"',
            f'{source_node} AND {target_node}',
            f'"{source_node}" AND "{target_node}" AND comorbidity',
            f'"{source_node}" AND "{target_node}" AND association',
            f'"{source_node}" AND "{target_node}" AND relationship',
            f'{source_node} AND {target_node} AND comorbid',
            f'{source_node} AND {target_node} AND co-occurring'
        ]
        return association_terms

    def download_papers_for_path(
        self, 
        path_nodes: List[str], 
        papers_per_node: int = DEFAULT_PAPERS_PER_NODE,
        max_total_papers: int = DEFAULT_MAX_TOTAL_PAPERS,
        include_intermediate: bool = True,
        include_pmc_metadata: bool = True,
        focus_on_associations: bool = True
    ) -> Dict[str, Any]:
        """
        Download papers for nodes in a shortest path, focusing on associations.
        
        Args:
            path_nodes: List of node names in the path
            papers_per_node: Number of papers to retrieve per node
            max_total_papers: Maximum total papers to download
            include_intermediate: Whether to include intermediate nodes
            include_pmc_metadata: Whether to fetch PMC IDs and URLs
            focus_on_associations: Whether to search for associations between nodes
            
        Returns:
            Dictionary with download results and summary
        """
        logger.info(f"Starting paper download for path: {' → '.join(path_nodes)}")
        
        all_papers = []
        summary = []
        seen_pmids = set()
        
        if focus_on_associations and len(path_nodes) >= 2:
            # Focus on associations between source and target
            source_node = path_nodes[0]
            target_node = path_nodes[-1]
            
            logger.info(f"Searching for associations between: {source_node} and {target_node}")
            
            # Create association search terms
            association_terms = self.create_association_search_terms(source_node, target_node)
            
            association_papers_found = 0
            
            for i, search_term in enumerate(association_terms):
                logger.info(f"Searching with term {i+1}: {search_term}")
                
                try:
                    # Fetch PMIDs for this association term
                    pmids = self.fetch_pmids_for_term(search_term, retmax=papers_per_node // len(association_terms))
                    
                    if not pmids:
                        summary.append({
                            'Search_Term': search_term,
                            'PMIDs_Found': 0,
                            'Papers_Processed': 0,
                            'Status': 'No PMIDs found'
                        })
                        continue
                
                    # Fetch paper details
                    papers = self.fetch_paper_details(pmids)
                    
                    # Fetch PMC metadata if requested
                    if include_pmc_metadata and papers:
                        pmc_data = self.fetch_pmc_metadata([p['pmid'] for p in papers])
                        
                        # Merge PMC metadata with paper data
                        for paper in papers:
                            pmid = paper['pmid']
                            if pmid in pmc_data:
                                pmc_info = pmc_data[pmid]
                                paper['pmc_id'] = pmc_info.get('pmc_id')
                                paper['pmc_url'] = pmc_info.get('pmc_url')
                            
                            # Use MeSH terms from PubMed only
                            paper['all_mesh_terms'] = paper.get('mesh_terms', [])
                    
                    # Filter out duplicates and add to results
                    term_papers = []
                    for paper in papers:
                        pmid = paper['pmid']
                        if pmid not in seen_pmids:
                            seen_pmids.add(pmid)
                            paper['search_term'] = search_term
                            paper['source_node'] = f"{source_node} → {target_node}"
                            term_papers.append(paper)
                            all_papers.append(paper)
                    
                    association_papers_found += len(term_papers)
                    
                    summary.append({
                        'Search_Term': search_term,
                        'PMIDs_Found': len(pmids),
                        'Papers_Processed': len(term_papers),
                        'Status': 'Success'
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing search term {search_term}: {e}")
                    summary.append({
                        'Search_Term': search_term,
                        'PMIDs_Found': 0,
                        'Papers_Processed': 0,
                        'Status': f'Error: {str(e)}'
                    })
            
            # If association search found very few papers, try individual node searches as fallback
            if association_papers_found < 5:
                logger.info(f"Association search found only {association_papers_found} papers. Trying individual node searches as fallback.")
                
                # Try individual searches for source and target
                for node in [source_node, target_node]:
                    logger.info(f"Fallback search for individual node: {node}")
                    
                    try:
                        pmids = self.fetch_pmids_for_term(node, retmax=papers_per_node // 2)
                        
                        if not pmids:
                            summary.append({
                                'Node': f"{node} (fallback)",
                                'PMIDs_Found': 0,
                                'Papers_Processed': 0,
                                'Status': 'No PMIDs found'
                            })
                            continue
                        
                        # Fetch paper details
                        papers = self.fetch_paper_details(pmids)
                        
                        # Fetch PMC metadata if requested
                        if include_pmc_metadata and papers:
                            pmc_data = self.fetch_pmc_metadata([p['pmid'] for p in papers])
                            
                            # Merge PMC metadata with paper data
                            for paper in papers:
                                pmid = paper['pmid']
                                if pmid in pmc_data:
                                    pmc_info = pmc_data[pmid]
                                    paper['pmc_id'] = pmc_info.get('pmc_id')
                                    paper['pmc_url'] = pmc_info.get('pmc_url')
                                
                                # Use MeSH terms from PubMed only
                                paper['all_mesh_terms'] = paper.get('mesh_terms', [])
                        
                        # Filter out duplicates and add to results
                        node_papers = []
                        for paper in papers:
                            pmid = paper['pmid']
                            if pmid not in seen_pmids:
                                seen_pmids.add(pmid)
                                paper['source_node'] = f"{node} (fallback)"
                                node_papers.append(paper)
                                all_papers.append(paper)
                        
                        summary.append({
                            'Node': f"{node} (fallback)",
                            'PMIDs_Found': len(pmids),
                            'Papers_Processed': len(node_papers),
                            'Status': 'Success'
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing fallback node {node}: {e}")
                        summary.append({
                            'Node': f"{node} (fallback)",
                            'PMIDs_Found': 0,
                            'Papers_Processed': 0,
                            'Status': f'Error: {str(e)}'
                        })
        else:
            # Fallback to individual node search
            if include_intermediate:
                nodes_to_process = path_nodes
            else:
                nodes_to_process = [path_nodes[0], path_nodes[-1]]
            
            for node in nodes_to_process:
                logger.info(f"Processing node: {node}")
                
                try:
                    # Fetch PMIDs for this node
                    pmids = self.fetch_pmids_for_term(node, retmax=papers_per_node)
                    
                    if not pmids:
                        summary.append({
                            'Node': node,
                            'PMIDs_Found': 0,
                            'Papers_Processed': 0,
                            'Status': 'No PMIDs found'
                        })
                        continue
                    
                    # Fetch paper details
                    papers = self.fetch_paper_details(pmids)
                    
                    # Fetch PMC metadata if requested
                    if include_pmc_metadata and papers:
                        pmc_data = self.fetch_pmc_metadata([p['pmid'] for p in papers])
                        
                        # Merge PMC metadata with paper data
                        for paper in papers:
                            pmid = paper['pmid']
                            if pmid in pmc_data:
                                pmc_info = pmc_data[pmid]
                                paper['pmc_id'] = pmc_info.get('pmc_id')
                                paper['pmc_url'] = pmc_info.get('pmc_url')
                            
                            # Use MeSH terms from PubMed only
                            paper['all_mesh_terms'] = paper.get('mesh_terms', [])
                    
                    # Filter out duplicates and add to results
                    node_papers = []
                    for paper in papers:
                        pmid = paper['pmid']
                        if pmid not in seen_pmids:
                            seen_pmids.add(pmid)
                            paper['source_node'] = node
                            node_papers.append(paper)
                            all_papers.append(paper)
                    
                    summary.append({
                        'Node': node,
                        'PMIDs_Found': len(pmids),
                        'Papers_Processed': len(node_papers),
                        'Status': 'Success'
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing node {node}: {e}")
                    summary.append({
                        'Node': node,
                        'PMIDs_Found': 0,
                        'Papers_Processed': 0,
                        'Status': f'Error: {str(e)}'
                    })
        
        # Limit total papers if needed
        if len(all_papers) > max_total_papers:
            # Sort by publication date (newest first) and take the best ones
            all_papers.sort(key=lambda x: x.get('publication_date', ''), reverse=True)
            all_papers = all_papers[:max_total_papers]
        
        logger.info(f"Downloaded {len(all_papers)} papers total")
        
        return {
            'success': True,
            'total_papers': len(all_papers),
            'summary': summary,
            'papers': all_papers,
            'path_nodes': path_nodes
        }
    
    def embed_papers_to_chromadb(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Embed downloaded papers into ChromaDB collection.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary with embedding results
        """
        if not papers:
            return {
                'success': False,
                'error': 'No papers to embed',
                'embedded_count': 0
            }
        
        try:
            logger.info(f"Embedding {len(papers)} papers into ChromaDB")
            
            # Prepare documents for embedding
            documents = []
            metadatas = []
            ids = []
            
            for paper in papers:
                # Create document text (title + abstract + MeSH terms)
                doc_parts = [paper.get('title', '')]
                
                if paper.get('abstract'):
                    doc_parts.append(paper.get('abstract'))
                
                # Add MeSH terms
                mesh_terms = paper.get('all_mesh_terms', paper.get('mesh_terms', []))
                if mesh_terms:
                    doc_parts.append(' '.join(mesh_terms))
                
                # Note: We only use abstracts, not full text
                
                document_text = ' '.join(doc_parts)
                
                # Prepare metadata
                metadata = {
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'journal': paper.get('journal', ''),
                    'publication_date': paper.get('publication_date', ''),
                    'authors': '; '.join(paper.get('authors', [])),
                    'mesh_terms': '; '.join(paper.get('mesh_terms', [])),
                    'all_mesh_terms': '; '.join(paper.get('all_mesh_terms', [])),
                    'source_node': paper.get('source_node', ''),
                    'pmc_id': paper.get('pmc_id', ''),
                    'pmc_url': paper.get('pmc_url', ''),
                    'source': 'path_downloader'
                }
                
                documents.append(document_text)
                metadatas.append(metadata)
                ids.append(paper['pmid'])
            
            # Add to ChromaDB collection
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully embedded {len(papers)} papers")
            
            return {
                'success': True,
                'embedded_count': len(papers),
                'collection_name': COLLECTION_NAME,
                'db_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"Error embedding papers: {e}")
            return {
                'success': False,
                'error': str(e),
                'embedded_count': 0
            }
    
    def save_results(self, results: Dict[str, Any], source_node: str, target_node: str) -> str:
        """Save download results to a JSON file."""
        try:
            # Create results directory
            results_dir = "path_papers_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_source = "".join([c if c.isalnum() or c in ' -_' else '_' for c in source_node]).strip()
            safe_target = "".join([c if c.isalnum() or c in ' -_' else '_' for c in target_node]).strip()
            filename = f"{safe_source}_to_{safe_target}_papers_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # Prepare data for saving
            save_data = {
                'metadata': {
                    'source_node': source_node,
                    'target_node': target_node,
                    'download_timestamp': timestamp,
                    'total_papers': results['total_papers'],
                    'success': results['success'],
                    'embedding_model': EMBEDDING_MODEL,
                    'collection_name': COLLECTION_NAME
                },
                'summary': results['summary'],
                'papers': results['papers']
            }
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': COLLECTION_NAME,
                'total_documents': count,
                'db_path': self.db_path,
                'embedding_model': EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'error': str(e)
            }

def test_simple_search():
    """Test simple PubMed search to verify connectivity."""
    downloader = PathPaperDownloader()
    
    # Test simple search
    print("Testing simple search...")
    pmids = downloader.fetch_pmids_for_term("depression", retmax=5)
    print(f"Found {len(pmids)} PMIDs for 'depression'")
    
    if pmids:
        print("Testing paper details fetch...")
        papers = downloader.fetch_paper_details(pmids[:2])
        print(f"Fetched details for {len(papers)} papers")
        if papers:
            print(f"Sample paper title: {papers[0].get('title', 'N/A')[:100]}...")
    
    return len(pmids) > 0

def main():
    """Example usage of the PathPaperDownloader."""
    # Test basic connectivity first
    if not test_simple_search():
        print("Basic search test failed. Check internet connection and NCBI access.")
        return
    
    # Example path nodes
    path_nodes = ["major depressive disorder", "bipolar disorder"]
    
    # Initialize downloader
    downloader = PathPaperDownloader()
    
    # Download papers
    results = downloader.download_papers_for_path(
        path_nodes=path_nodes,
        papers_per_node=5,
        max_total_papers=15,
        include_intermediate=True,
        include_pmc_metadata=True,
        focus_on_associations=True
    )
    
    if results['success']:
        print(f"Downloaded {results['total_papers']} papers")
        
        # Show summary
        for item in results['summary']:
            print(f"  {item}")
        
        if results['total_papers'] > 0:
            # Embed into ChromaDB
            embed_results = downloader.embed_papers_to_chromadb(results['papers'])
            
            if embed_results['success']:
                print(f"Embedded {embed_results['embedded_count']} papers into ChromaDB")
            
            # Save results
            filepath = downloader.save_results(results, path_nodes[0], path_nodes[-1])
            print(f"Results saved to: {filepath}")
            
            # Show collection stats
            stats = downloader.get_collection_stats()
            print(f"Collection stats: {stats}")
        else:
            print("No papers found to embed.")
    else:
        print(f"Download failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
