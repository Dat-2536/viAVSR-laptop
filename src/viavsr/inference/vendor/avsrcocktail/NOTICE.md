# Vendored AVSRCocktail model code

The Python files in this directory are the minimal dependency closure needed to
instantiate the released Vietnamese AV-HuBERT CTC/Attention checkpoint and
reproduce its joint CTC/attention beam-search decoder. They
were derived from:

- Repository: https://github.com/nguyenvulebinh/AVSRCocktail
- Commit: `51107b66864c42687638a00df8dd398ec9210872`
- Upstream license: Creative Commons Attribution-NonCommercial 4.0

Changes in this project are limited to package-relative imports, removal of
unused model families, replacement of removed NumPy scalar aliases,
modernization of Python type/style syntax, and a small load-only wrapper.
Dataset, training, and English-tokenizer code are intentionally excluded.

The beam-search files retain their upstream Apache-2.0 attribution headers and
are wired to the Vietnamese unigram2048 token list by project-owned code.
