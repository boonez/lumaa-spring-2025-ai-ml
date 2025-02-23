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
2. WALL·E (Score: 6.1273)
3. The Right Stuff (Score: 5.7932)
4. The Iron Giant (Score: 5.7181)
5. Interstellar (Score: 5.0697)
```

```bash
py recommend.py "I like Chris Evans movies" "imdb_top_1000.csv"        
Top recommendations:
1. Knives Out (Score: 6.9483)
2. Gifted (Score: 6.8000)
3. Avengers: Endgame (Score: 5.0807)
4. The Avengers (Score: 4.9666)
5. Captain America: Civil War (Score: 4.9247)
```

```bash
py recommend.py "romantic" "imdb_top_1000.csv"
Top recommendations:
1. Call Me by Your Name (Score: 8.6000)
2. The Apartment (Score: 7.7074)
3. Jules et Jim (Score: 7.4763)
4. Gone with the Wind (Score: 6.6945)
5. Charade (Score: 6.6389)
```

```bash
py recommend.py "I am looking for romance movies" "imdb_top_1000.csv" 
Top recommendations:
1. Call Me by Your Name (Score: 8.6000)
2. The Apartment (Score: 7.7099)
3. Jules et Jim (Score: 7.4902)
4. Gone with the Wind (Score: 6.8877)
5. Charade (Score: 6.1570)
```
## Salary Expectations
 $4000 a month

