# RAG with LangChain and Chroma

## Prerequisites

- Install [Ollama](https://ollama.com/download)

- Create virtual environment

- This was tested with python 3.10

```bash
python -m venv venv
source venv/bin/activate # On Windows use `source venv/Scripts/Activate`
pip3 install -r requirements.txt
# torch is needed for chunking purposes under the hood
pip3 install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121 # for some reason it's not installed properly via requirements.txt
```

## Setup the database

```bash
python database.py --type=markdown #chunk markdown files
python database.py --type=pdf #chunk pdf files
python database.py --reset # Clear the database
```

## USAGE

### Start the chat app

```bash
python -m chainlit run chat.py -w
```

Visit: <http://localhost:8000>

### Do one time query via command line

```bash
python query_data.py "how to create admin user via cli?"
```
