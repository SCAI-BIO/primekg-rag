#!/usr/bin/env python3
"""
Database Setup Script for PrimeKG RAG
Downloads and extracts databases from Zenodo repository
"""

import os
import sys
import requests
import py7zr
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ZENODO_URL = "https://zenodo.org/records/17121721/files/databases.7z?download=1"
DATABASE_ARCHIVE = "databases.7z"
EXTRACT_DIR = Path(__file__).parent

def download_databases():
    """Download the databases archive from Zenodo"""
    logger.info(f"Downloading databases from {ZENODO_URL}")
    logger.info("This will download a large file (~2GB) from Zenodo record 17121721")
    logger.info("URL: https://zenodo.org/records/17121721")
    
    try:
        response = requests.get(ZENODO_URL, stream=True)
        response.raise_for_status()
        
        # Verify we're getting data from Zenodo
        if 'zenodo.org' not in response.url:
            raise Exception("Redirect away from Zenodo detected")
            
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            raise Exception("No content length received from Zenodo")
            
        logger.info(f"Expected download size: {total_size / (1024*1024):.1f} MB")
        
        downloaded = 0
        
        with open(DATABASE_ARCHIVE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Successfully downloaded {DATABASE_ARCHIVE} ({downloaded / (1024*1024):.1f} MB)")
        # Log file size
        logger.info(f"Downloaded archive size: {os.path.getsize(DATABASE_ARCHIVE) / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download databases: {e}")
        return False

def extract_databases():
    """Extract the databases archive"""
    logger.info(f"Extracting {DATABASE_ARCHIVE}")
    
    try:
        # Create directories if they don't exist
        for dir_name in ["pubmed_db", "node_db", "question_db", "shortest_path_db", "new_subgraphs"]:
            dir_path = EXTRACT_DIR / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Ensuring directory exists: {dir_path}")

        with py7zr.SevenZipFile(DATABASE_ARCHIVE, mode='r') as archive:
            # Log archive contents before extraction
            logger.info(f"Archive contains: {archive.getnames()}")
            logger.info(f"Extracting to: {EXTRACT_DIR}")
            archive.extractall(path=EXTRACT_DIR)
        
        # Verify extracted contents immediately
        extracted_files = []
        for dirpath, _, filenames in os.walk(EXTRACT_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                size = os.path.getsize(filepath)
                extracted_files.append(f"{filepath} ({size/1024/1024:.2f} MB)")
        
        logger.info("Extracted files:")
        for f in extracted_files:
            logger.info(f"  {f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract databases: {e}")
        return False

def cleanup():
    """Remove the archive file after extraction"""
    try:
        if os.path.exists(DATABASE_ARCHIVE):
            os.remove(DATABASE_ARCHIVE)
            logger.info(f"Removed {DATABASE_ARCHIVE}")
    except Exception as e:
        logger.warning(f"Could not remove {DATABASE_ARCHIVE}: {e}")

def verify_databases():
    """Verify that required database directories exist and are not empty (ignore hidden files)"""
    required_dirs = [
        "pubmed_db",
        "node_db", 
        "question_db",
        "shortest_path_db",
        "new_subgraphs"
    ]
    missing_dirs = []
    empty_dirs = []
    for dir_name in required_dirs:
        dir_path = EXTRACT_DIR / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        else:
            # Only count non-hidden files
            real_files = [f for f in dir_path.iterdir() if not f.name.startswith('.')]
            logger.info(f"Files in {dir_name}: {[f.name for f in real_files]}")
            if not real_files:
                empty_dirs.append(dir_name)
    if missing_dirs:
        logger.warning(f"Missing database directories: {missing_dirs}")
    if empty_dirs:
        logger.warning(f"Empty database directories (no real data files): {empty_dirs}")
    if missing_dirs or empty_dirs:
        return False
    logger.info("All required database directories found and non-empty")
    return True

def main():
    logger.info(f"Running in directory: {os.getcwd()}")
    logger.info("Starting database setup...")
    
    # Always download fresh copy
    if not download_databases():
        return False
    if not extract_databases():
        return False
    cleanup()
    
    # Verify final setup
    if verify_databases():
        logger.info("Database setup completed successfully!")
        return True
    else:
        logger.error("Database setup completed but verification failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
