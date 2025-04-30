# Taxotron
A Taxonomy Enrichment Demo

This repository demonstrates how to enrich a simple taxonomy using an LLM. The script takes an initial taxonomy (in JSON format) and enriches it by adding new categories and subcategories, leveraging LLM capabilities.

## Getting Started

To get started, follow the instructions below.

### 1. **Clone the repository**

First, you'll need to clone this repository to your local machine so that you can run and modify the code.

```bash
git clone https://github.com/zeinabGhamlouch/taxotron.git
cd taxotron
```

### 2. **Install dependencies**
Next, make sure you have all the necessary dependencies installed by running the following command. This will install everything listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. **Run the demo**
Once everything is set up, you can run the demo to enrich the taxonomy. This will use the sample `example_taxonomy.json` file as input and generate an enriched version:

```bash
python demo/run_demo.py
```

After running this command, the enriched taxonomy will be saved to `demo_example/example_taxonomy_enriched.json`.

## 4. Examples
In the folder names `examples` you can find the taxonomies we worked with and result we got by enriching each one with different LLMs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



