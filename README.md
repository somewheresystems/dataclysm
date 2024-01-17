```
@@@@@%%%%%@#-..:-*%%@@@@@@#********%@@@@%%%%#*++#@#*#@%%@@@%%%%%%%%%%%%%%%%+..:*%#%%%%@@@@
@@@@@%%%%##@@=-*=..+%#+::+@-:-:::.*#-:-%%%*:.---.@*.#%@#:.%*.:#-.---=%--%@*..-%@##%%%%@@@@
@@@@@%%%%#*%@+-#@#::*@*---@*@@-+@%*%:=.#%+:=%@@@#@#:#%@@%=:.-@#:-%@@*%=:+%-.-=@#*#%%%%@@@@
@@@@@%%%%##%@+-#@%=-*@-+*.#@@@-+@@@+-%-:@=-*@@@@@@#:#@@@@%-:@@@%+-=#@%=-:=.+=+@#*#%%%%@@@@
@@@@@%%%%##%@*+%@#=+%+-==-+@@@=+@@%-===-*#==%@@@%@#=#@@@%%=+@@%@@@*-*@+**-=%+*@###%%%%@@@@
@@@@@%%%%##@@#+#=-*%%=*@@#=%@@=+@@+=%@@+=@%+:-===@#-===-#@++@@#-++-:*@+#@+%@*#@@*#%%%%@@@@
@@@@@%%%%##%@++*%@@@#%#@@#@#@%@%%%%%%@%%%%@@@%%@@@#@@@@@@%%%%@#@%#%@@%%#@@@%*+@%*#%%%%@@@@
```
This repository provides a comprehensive guide to using DATACLYSM: a series of high-quality embeddings libraries, with coverage for the entirety of English Wikipedia and arXiv. The guide is based on the `getting_started.ipynb` notebook.

## Table of Contents
1. [Installation](#installation)
2. [Initialization](#initialization)
3. [Retrieval Augmented Generation](#retrieval-augmented-generation)
4. [Reranking Results](#reranking-results)
5. [License](#license)

## Installation
To install the necessary dependencies, run the following command:
```python
%pip install -r requirements.txt
```

## Initialization
The initialization process involves setting up the Wikipedia Database and Index. This process takes approximately 12 minutes to index on an M3 Max.
```python
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagModel
```

## Retrieval Augmented Generation
The Retrieval Augmented Generation (RAG) process uses a model to encode a query and retrieve examples based on title similarity. The examples are then converted to a DataFrame and displayed.

## Reranking Results
The reranking process uses an LLM to return a list instructing the LLM to rerank and drop irrelevant results. The results are then displayed as a table with hyperlinks.

## License
This project is licensed under the Apache License 2.0. For more details, see the `LICENSE` file in the repository.

For more detailed instructions and examples, refer to the `getting_started.ipynb` notebook.
