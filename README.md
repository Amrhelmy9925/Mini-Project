pip install -e ".[dev]" # install
ruff check src tests && black --check src tests # lint
pytest -v -o addopts="" # test
python -m prodml.train # train
uvicorn prodml.api.main:app --reload --port 8000 # serve