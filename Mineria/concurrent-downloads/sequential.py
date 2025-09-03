import os
import typing as t

import requests

import utils


def download_and_save_pokemon(session, pokemon, output_dir):
    """Download and save a single pokemon."""
    content = utils.maybe_download_sprite(session, pokemon["Sprite"])
    if content is not None:
        target_dir = os.path.join(output_dir, pokemon["Type1"])
        utils.maybe_create_dir(target_dir)
        filepath = os.path.join(target_dir, pokemon["Pokemon"] + ".png")
        utils.write_binary(filepath, content)


def dowload_and_save_all_pokemons(pokemons, output_dir):
    """Download and save all pokemons using a sequentially."""
    with requests.Session() as session:
        for p in pokemons:
            download_and_save_pokemon(session, p, output_dir)

@utils.timeit
def main(output_dir: str, inputs: t.List[str]):
    """Download for all intpus and place them in output_dir."""
    utils.maybe_create_dir(output_dir)
    dowload_and_save_all_pokemons(utils.read_pokemons(inputs), output_dir)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="directory to store the data")
    parser.add_argument("inputs", nargs="+", help="list of files with metadata")
    args = parser.parse_args()
    utils.maybe_remove_dir(args.output_dir)
    main(args.output_dir, args.inputs)