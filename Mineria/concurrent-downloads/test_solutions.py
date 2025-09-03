import os
import shutil
import unittest

import pandas as pd

import utils
from asyncio_ import main as asyncio_main
from sequential import main as sync_main
from threading_ import main as threading_main
from multiprocessing_ import main as multiprocessing_main


def _find_all_datasets():
    paths = [f"data/pokemon-gen{i}-data.csv" for i in range(1, 7)]
    return paths
        
def _load_combined_dataframe():
    all_dfs = []
    for path in _find_all_datasets():
        df = pd.read_csv(path)
        df["Pokemon"] = df["Pokemon"].str.lower()
        df["Type1"] = df["Type1"].str.lower()
        all_dfs.append(df)
    return pd.concat(all_dfs)


def _test_correctness(output_dir):
    df = _load_combined_dataframe()
    cross = pd.crosstab(df["Type1"], df["Pokemon"])
    categories = {c for c in cross.index}
    for root, dirs, files in os.walk(output_dir):
        if root == output_dir:
            assert categories == set(dirs)
            assert files == []
        else:
            category_name = os.path.split(root)[-1]
            pokemons_by_category = cross.loc[category_name]
            expected_names = {f"{n}.png" for n in pokemons_by_category[pokemons_by_category != 0].index}
            assert expected_names == set(files)
            assert dirs == []

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.inputs = _find_all_datasets()
        self.output_dir = "testing_output"
        utils.maybe_remove_dir(self.output_dir)
        utils.maybe_create_dir(self.output_dir)

    def tearDown(self):
        utils.maybe_remove_dir(self.output_dir)

    def test_synchronous(self):
        sync_main(self.output_dir, self.inputs)
        _test_correctness(self.output_dir)
        utils.maybe_remove_dir(self.output_dir)

    def test_asyncio(self):
        asyncio_main(self.output_dir, self.inputs)
        _test_correctness(self.output_dir)
        utils.maybe_remove_dir(self.output_dir)

    def test_multiprocessing(self):
        multiprocessing_main(self.output_dir, self.inputs)
        _test_correctness(self.output_dir)
        utils.maybe_remove_dir(self.output_dir)

    def test_threading(self):
        threading_main(self.output_dir, self.inputs)
        _test_correctness(self.output_dir)
        utils.maybe_remove_dir(self.output_dir)