# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

## Dataset
Download [Top 1000 IMDB movies and TV shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) csv file from Kaggle

## Setup
- Python version 3.12.6 
- Requirements: py -m pip install -r requirements.txt
    - pandas
    - numpy
    - scikit-learn
## Running
```bash
py recommend.py "Some user description" "path/to/dataset.csv"
```
Example
```bash
py recommend.py "I like adventure movies in space" "imdb_top_1000.csv"
```
## Results 

```bash
py recommend.py "I like adventure movies in space" "imdb_top_1000.csv"
Top recommendations:
1. Aliens (Score: 8.3500)
2. WALL·E (Score: 5.8873)
3. The Right Stuff (Score: 5.7730)
4. The Iron Giant (Score: 5.6767)
5. Interstellar (Score: 5.1243)
```

```bash
py recommend.py "I like Chris Evans movies" "imdb_top_1000.csv"
Top recommendations:
1. Captain America: Civil War (Score: 7.6001)
2. Knives Out (Score: 7.4684)
3. Gifted (Score: 6.8000)
4. Avengers: Endgame (Score: 6.4897)
5. The Avengers (Score: 6.4784)
```

```bash
py recommend.py "romantic" "imdb_top_1000.csv"
Top recommendations:
1. Call Me by Your Name (Score: 8.6000)
2. The Apartment (Score: 7.7089)
3. Jules et Jim (Score: 7.4424)
4. Gone with the Wind (Score: 6.8313)
5. Portrait de la jeune fille en feu (Score: 6.1934)
```

```bash
py recommend.py "I am looking for romance movies" "imdb_top_1000.csv"    
Top recommendations:
1. Call Me by Your Name (Score: 8.6000)
2. The Apartment (Score: 7.7408)
3. Jules et Jim (Score: 7.4474)
4. Gone with the Wind (Score: 6.7358)
5. Charade (Score: 6.2661)
```
## Salary Expectations
 $4000 a month

