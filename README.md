# Running locally

## Prerequisites
- Python 3.8 or higher
- Git

## Complete Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/SCAI-BIO/primekg-rag.git
cd primekg-rag
```

2. **Create and activate a Python virtual environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Git Bash):
source venv/Scripts/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment file:**
```bash
# Create .env file with correct paths
echo "OPENAI_API_KEY=your_key_here" > .env
echo "PUBMED_DB_PATH=pubmed_db" >> .env
```
*Replace `your_key_here` with your actual OpenAI API key*

5. **Download and set up databases:**
```bash
cd primekg-rag
python setup_databases.py
```
*This will download ~450MB of data including CSV files and database collections*

6. **Run the application:**
```bash
# Make sure you're in the primekg-rag directory
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Troubleshooting

### Common Issues:

**"streamlit: command not found"**
- Make sure the virtual environment is activated: `source venv/Scripts/activate`

**"PUBMED_DB_PATH not set in .env file"**
- Ensure the .env file is in the root directory (not in primekg-rag/)
- Check that PUBMED_DB_PATH=pubmed_db (not primekg-rag/pubmed_db)

**"Collection [pubmed_abstracts] does not exist"**
- Run the database setup again: `cd primekg-rag && python setup_databases.py`
- Verify the .env file has the correct PUBMED_DB_PATH

**Missing CSV files warnings**
- The setup script should download nodes.csv and best_question_matches.csv automatically
- If missing, re-run: `cd primekg-rag && python setup_databases.py`

# Running via Docker

## Prerequisites
- Docker and Docker Compose installed
- Git

## Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/SCAI-BIO/primekg-rag.git
cd primekg-rag
```

2. **Set up environment file:**
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
echo "PUBMED_DB_PATH=primekg-rag/pubmed_db" >> .env
```
*Replace `your_key_here` with your actual OpenAI API key*

3. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8501`

## Additional Notes

- Make sure to replace `your_key_here` with your actual OpenAI API key in the `.env` file
- The first run may take longer as it downloads and sets up the required databases
- For development, you may want to use the local setup instead of Docker for faster iteration
