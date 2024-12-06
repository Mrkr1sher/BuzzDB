# Quickstart

## Prerequisites
- C++ compiler with C++17 support

## Dataset Setup
1. Download the GIST-1M dataset from [here](http://corpus-texmex.irisa.fr/)
2. Create a `gist` directory in your project root
3. Extract and place these files in the `gist` directory:
   - `gist/gist_base.fvecs` (base vectors)
   - `gist/gist_query.fvecs` (query vectors)

## Build & Run Tests
```bash
g++ -g -std=c++17 buzzdb.cpp -o buzzdb
./buzzdb
```

## Run Evaluation
```bash
g++ -std=c++17 -o evaluation evaluation.cpp
./evaluation
```

