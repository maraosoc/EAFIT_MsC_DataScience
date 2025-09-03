# concurrent-downloads

In this homework you are required to build scripts to automatically download images files following three approaches:

- `asyncio`
- `multiprocessing`
- `threading`

If you still do not know about what those are, please refer to [Speed Up Your Python Programm With Concurrency](https://realpython.com/python-concurrency/).

You will find `sequential.py` already implemented the repository, which has the code requried to execute the downloads sequentialy. Your task is to create concurrent versions of that program. 


You will be downloading pokemons, using the dataset taken from dataset was taken from [Pokemon Shwodown Webscraper](https://github.com/travishn/pokemon-showdown-scraper). This dataset consist of several CSV files similar to the follwing:


| Pokemon    | Number | Type1    | Type2    | Ability1      | Ability2      | Ability3      | HP  | Attack | Defense | Sp. Atk | Sp. Def | Speed | Sprite                                                     |
|------------|--------|----------|----------|---------------|---------------|---------------|-----|--------|---------|---------|---------|-------|------------------------------------------------------------|
| Bulbasaur  | 1      | GRASS    | POISON   | Overgrow      | Chlorophyll   | none          | 45  | 49     | 49      | 65      | 65      | 45    | https://play.pokemonshowdown.com/sprites/bw/bulbasaur.png  |
| Ivysaur    | 2      | GRASS    | POISON   | Overgrow      | Chlorophyll   | none          | 60  | 62     | 63      | 80      | 80      | 60    | https://play.pokemonshowdown.com/sprites/bw/ivysaur.png    |
| Venusaur   | 3      | GRASS    | POISON   | Overgrow      | Chlorophyll   | none          | 80  | 82     | 83      | 100     | 100     | 80    | https://play.pokemonshowdown.com/sprites/bw/venusaur.png   |
| Charmander | 4      | FIRE     | none     | Blaze         | Solar Power   | none          | 39  | 52     | 43      | 60      | 50      | 65    | https://play.pokemonshowdown.com/sprites/bw/charmander.png |
| Charmeleon | 5      | FIRE     | none     | Blaze         | Solar Power   | none          | 58  | 64     | 58      | 80      | 65      | 80    | https://play.pokemonshowdown.com/sprites/bw/charmeleon.png |


Each image should be stored in a directory with the name of the type (`Type1`) of the pokemon and the image stored with the name of the pokemon in PNG format as follows:

```
output
│
└───grass
│   │   bulbasaur.png
│   │   ivysaur.png
│   │   ...
│   
└───fire
    │   charmander.png
    │   charmeleon.png
    |   ...
```

You are given a set of functions in `utils.py`that you are free to use if you find them useful, but you are free to structure your code in any way. The only requirement is that the files `asyncio_.py`, `threading_.py` and `multiprocessing_.py` all implemented a function called main that accept the output directory and the filepath of all CSV files that make up the dataset.

If you follow the [tutorial mentioned above](https://realpython.com/python-concurrency/), it is very likely that the execution speed of the implementations follows the following rank:

1. `asyncio`
2. `threading`
3. `multiprocessing`
4. `sequential`

However, the exact ranking can vary according to your implementation and your machine. For instance, it could happend that the threading version takes about the same as the `asyncio` version. 

Please **DO NOT** make a PR with the solutions!