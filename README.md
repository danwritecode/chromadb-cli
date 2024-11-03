# Chromadb CLI
This tool is entirely written by claude sonnet 3.6. 

Create venv with compatible version: 
```
python3.12 -m venv venv
source venv/bin/activate
```

You just need to create a .env file with the following items: 
```
CHROMA_HOST=
CHROMA_PORT=8000
CHROMA_SSL=false
```

## Commands
```
# list all collections
python main.py list

# create a new collection
python main.py create test_collection --distance cosine

# peek at contents
python main.py peek test_collection --limit 5

# quick semantic search 
python main.py search test_collection "what is machine learning" --n-results 3

# get collection stats
python main.py stats test_collection

# delete collection
python main.py delete test_collection

# help
python main.py --help

# command help
python main.py create --help
python main.py search --help
```
