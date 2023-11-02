# FastAPI Embedding API

## Overview

The FastAPI Embedding API provides a simple interface for generating embedding vectors from input sentences. Embedding vectors are numerical representations of text that capture semantic information, allowing for various NLP tasks.

## Getting Started

### Installation

1. Make sure you have Python 3.x installed on your system.
2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Running the API

To start the FastAPI server, run the following command:

```bash
uvicorn main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`.

## Usage

Send a POST request to `http://127.0.0.1:8000/embed` with a JSON object containing a list of sentences.

### Example Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/embed' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "This is a sample sentence.",
    "Another example for embedding."
  ]
}'
```

### Example Response

```json
[
    {
        "sentence": "This is a sample sentence.",
        "embedding": [
            -0.30739638209342957,
            -0.3990561366081238,
            -0.6577262878417969,
            0.2739684283733368,
            -0.09149201959371567,
            0.2765969932079315,
            -0.010574031621217728,
            -0.9837483167648315,
            -0.9544899463653564,
            1.1655793190002441,
            0.4724787175655365,
            -0.8983410596847534,
            -0.4408877193927765,
            0.5333910584449768,
            0.30021241307258606,
            -0.7057181596755981,
            0.20055779814720154,
            -0.7414146661758423,
            -0.16691547632217407,
            -0.7831200361251831,
            0.2316111922264099,
            0.3517184257507324,
            -0.49855148792266846,
            0.3873681426048279,
            0.8929033279418945,
            -0.3907497525215149,
            -0.20189079642295837,
            -0.27217328548431396,
            -0.20150843262672424,
            -0.2696434557437897,
            -0.04416860267519951,
            -0.1778184473514557
        ]
    },
    {
        "sentence": "Another example for embedding.",
        "embedding": [
            -0.11383337527513504,
            -0.21959248185157776,
            -0.2680359482765198,
            0.2703028917312622,
            0.13453321158885956,
            0.3442433178424835,
            -0.641788125038147,
            -0.4218224287033081,
            -0.9396926760673523,
            -0.1322430521249771,
            0.825551450252533,
            0.362534761428833,
            -0.15542413294315338,
            -0.12511911988258362,
            -0.1556849330663681,
            0.13272254168987274,
            0.2515423595905304,
            -0.6910218596458435,
            0.16325613856315613,
            -0.41506609320640564,
            0.31772831082344055,
            0.14017413556575775,
            -0.10088400542736053,
            0.2694193124771118,
            0.1680845022201538,
            -0.4719347655773163,
            0.015851259231567383,
            -0.3199581801891327,
            -0.07567527890205383,
            -0.11492110788822174,
            -0.007399208843708038,
            0.29920971393585205
        ]
    }
]
```
## Acknowledgments

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

Please remember to replace placeholders like `http://127.0.0.1:8000` with your actual API endpoint if needed. Additionally, include any additional sections or information that is specific to your project.