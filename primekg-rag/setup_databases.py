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
ZENODO_URL = "https://zenodo.org/records/17119877"
DATABASE_ARCHIVE = "databases.7z"
EXTRACT_DIR = Path(__file__).parent

def download_databases():
    """Download the databases archive from Zenodo"""
    logger.info(f"Downloading databases from {ZENODO_URL}")
    
    try:
        response = requests.get(ZENODO_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
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
        return True
        
    except Exception as e:
        logger.error(f"Failed to download databases: {e}")
        return False

def extract_databases():
    """Extract the databases archive"""
    logger.info(f"Extracting {DATABASE_ARCHIVE}")
    
    try:
        with py7zr.SevenZipFile(DATABASE_ARCHIVE, mode='r') as archive:
            archive.extractall(path=EXTRACT_DIR)
        
        logger.info("Successfully extracted databases")
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
    """Verify that required database directories exist"""
    required_dirs = [
        "pubmed_db",
        "node_db", 
        "question_db",
        "shortest_path_db",
        "new_subgraphs"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = EXTRACT_DIR / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        logger.warning(f"Missing database directories: {missing_dirs}")
        return False
    
    logger.info("All required database directories found")
    return True

def main():
    """Main setup function"""
    logger.info("Starting database setup...")
    
    # Check if databases already exist
    if verify_databases():
        logger.info("Databases already exist, skipping download")
        return True
    
    # Download and extract databases
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
