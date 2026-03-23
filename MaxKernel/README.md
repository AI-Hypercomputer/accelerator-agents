# tpu_kernel_gen


## Installation

### As a Package (Recommended)

Install this package locally for use anywhere:

```bash
# From the project directory
pip install -e .
```

This allows you to import and use the modules from anywhere:

```python
from tpu_kernel_gen.kernel_parser import parse_kernels
from tpu_kernel_gen.embed import generate_embeddings
from tpu_kernel_gen.kernel_retrieval import search_similar_kernels
```

### Dependencies Only

Alternatively, install just the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### As Python Package

After installing the package, you can use it programmatically:

```python
import tpu_kernel_gen.kernel_parser as parser
import tpu_kernel_gen.embed as embedder
import tpu_kernel_gen.kernel_retrieval as retriever

# Parse kernels
kernels = parser.parse_kernels("/path/to/source")

# Generate embeddings
embedder.add_embeddings("kernels.csv")

# Search for similar kernels
results = retriever.search_kernels("matrix multiplication", k=10)
```

### Command Line Usage

## How to populate kernel DB
### Step 1: Parse kernels from source code

Use the kernel parser to extract Pallas kernels from Python source files:

```bash
python kernel_parser.py /path/to/source/directory --output kernels.csv
```

This will:
- Recursively scan Python files for JAX Pallas kernels
- Extract kernel definitions and call sites
- Save results to `kernels.csv`

### Step 2: Generate embeddings

Add code embeddings to the kernel data using UniXcoder:

```bash
python embed.py kernels.csv --code_column code
```

This will:
- Load the UniXcoder model for code embeddings
- Process each kernel's code to generate vector embeddings
- Add embedding columns to the CSV file in-place
- Create a backup of the original file

### Step 3: Upload to BigQuery

Upload the enriched kernel data to BigQuery:

```bash
python bq_upload.py --csv-file kernels.csv --table-name your_dataset.kernels --project-id your-project-id
```

This will:
- Upload the CSV data to the specified BigQuery table
- Auto-generate incremental UUIDs for new entries
- Apply the proper schema for the kernel database


## How to retrieve from kernel DB


Use the kernel retrieval tool to search for similar kernels in the BigQuery vector database:

```bash
python kernel_retrieval.py --project-id your-project-id --dataset-name your_dataset --table-name kernels --query "matrix multiplication kernel" --k 10
```

This will:
- Connect to your BigQuery vector store using UniXcoder embeddings
- Search for kernels similar to your query using cosine similarity
- Return the top k most similar results with metadata and similarity scores
- Display operation names, frameworks, hardware targets, and file locations

Optional flags:
- `--verbose`: Enable detailed output during the search process
- `--k`: Number of similar kernels to retrieve (default: 5)

The results will show ranked kernels with their similarity scores, operation metadata, and source file information to help you find relevant kernel implementations.
