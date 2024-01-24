```
@@@@@%%%%%@#-..:-*%%@@@@@@#********%@@@@%%%%#*++#@#*#@%%@@@%%%%%%%%%%%%%%%%+..:*%#%%%%@@@@
@@@@@%%%%##@@=-*=..+%#+::+@-:-:::.*#-:-%%%*:.---.@*.#%@#:.%*.:#-.---=%--%@*..-%@##%%%%@@@@
@@@@@%%%%#*%@+-#@#::*@*---@*@@-+@%*%:=.#%+:=%@@@#@#:#%@@%=:.-@#:-%@@*%=:+%-.-=@#*#%%%%@@@@
@@@@@%%%%##%@+-#@%=-*@-+*.#@@@-+@@@+-%-:@=-*@@@@@@#:#@@@@%-:@@@%+-=#@%=-:=.+=+@#*#%%%%@@@@
@@@@@%%%%##%@*+%@#=+%+-==-+@@@=+@@%-===-*#==%@@@%@#=#@@@%%=+@@%@@@*-*@+**-=%+*@###%%%%@@@@
@@@@@%%%%##@@#+#=-*%%=*@@#=%@@=+@@+=%@@+=@%+:-===@#-===-#@++@@#-++-:*@+#@+%@*#@@*#%%%%@@@@
@@@@@%%%%##%@++*%@@@#%#@@#@#@%@%%%%%%@%%%%@@@%%@@@#@@@@@@%%%%@#@%#%@@%%#@@@%*+@%*#%%%%@@@@
```
This repository provides a comprehensive guide to getting started with using DATACLYSM: a series of high-quality embeddings libraries, with coverage for the entirety of PubMed, English Wikipedia and arXiv. The guide is based on the `getting_started.ipynb` notebook.

It also includes a demo of the Spatial Search Engine, a Streamlit app for exploring the Dataclysm datasets visually and performing ranked searches on proximally related articles (by title, currently).

## Table of Contents
1. [Installation](#installation)
2. [Initialization](#initialization)
3. [Retrieval Augmented Generation](#retrieval-augmented-generation)
4. [Reranking Results](#reranking-results)
5. [License](#license)

## Installation
To install the necessary dependencies, run the following command in a fresh conda environment. I suggest Python 3.10:
```python
%pip install -r requirements.txt
```

## Retrieval Augmented Generation
The Retrieval Augmented Generation (RAG) demonstration uses the `BAAI/bge_small_en_v2` model to encode a query and retrieve examples based on title similarity using FAISS. The examples are then summarized using Hermes-2.5-Mistral-7B.

## Reranking Results
Demos are included for classical (score augmentation) and LLM-based (experimental) reranking of results. The experimental LLM reranking process uses the aforementioned model to return a list instructing the LLM to rerank and drop irrelevant results. The results are then displayed as a table with hyperlinks.

## Streamlit SSE (Spatial Search Engine) Demo
To run the Streamlit demo, simply navigate to the demo directory and run the Streamlit app:
```bash
cd streamlit-demo
streamlit run app.py
```

## License
This project is licensed under the Apache License 2.0. For more details, see the `LICENSE` file in the repository.

For more detailed instructions and examples, refer to the `getting_started.ipynb` notebook.
