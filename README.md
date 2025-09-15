# Running locally

setting up env file

```bash
git clone https://github.com/SCAI-BIO/primekg-rag.git
cd primekg-rag
echo "OPENAI_API_KEY=your_key_here" > .env
pip install -r requirements.txt
#unzip databases
python setup_databases.py
streamlit run app.py

```

# Running via docker

```bash
git clone https://github.com/SCAI-BIO/primekg-rag.git
cd primekg-rag
echo "OPENAI_API_KEY=your_key_here" > .env
docker-compose up --build


```
