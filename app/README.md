# personIA 

Este projeto oferece uma API que recebe um trecho de um livro e retorna as características (traços) de um personagem inferidas por um modelo de IA já treinado.

Principais pontos
- Entrada: texto (trecho de livro)
- Saída: schema JSON com labels e scores. Exemplo:
  {
    "labels": ["protetor", "determinado"],
    "scores": {"protetor": 0.7141908960625957, "determinado": 0.2183861662956591}
  }
- O modelo é carregado a partir de: `data/traits_pipeline.joblib`

Requisitos
- Python 3.10+ (recomendado)
- Recomenda-se criar um virtualenv

Instalação usando [uv](https://docs.astral.sh/uv/)

1. Criar e ativar virtualenv

```powershell
uv new -p python3.12 .venv
```

2. Instalar dependências

```powershell
uv sync
```

Rodando a API localmente

```powershell
# a partir da raiz do repositório
$ cd app
$ uv run fastapi run dev main.py
```

A API deverá estar disponível em: http://127.0.0.1:8000

Endpoint principal
- POST /api/v1/prediction/analyze
  - Body (application/json): { "text": "<trecho do livro>" }
  - Response (application/json): schema mostrado acima
