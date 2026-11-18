
import subprocess
import sys

from config import SEEDS


def main():

    print("=" * 60)
    print("MULTI SEED PPO + A2C + DQN EXPERIMENT")
    print("=" * 60)

    algorithms = [
        "PPO",
        "A2C",
        "DQN"
    ]

    for algorithm in algorithms:

        print()
        print("=" * 60)
        print(f"STARTING {algorithm} EXPERIMENT")
        print("=" * 60)

        for seed in SEEDS:

            print()
            print("=" * 60)
            print(f"Training {algorithm} with seed {seed}")
            print("=" * 60)

            if algorithm == "PPO":

                train_module = "src.train_ppo"

            elif algorithm == "A2C":

                train_module = "src.train_a2c"

            elif algorithm == "DQN":

                train_module = "src.train_dqn"

            else:

                continue

            result = subprocess.run(

                [
                    sys.executable,
                    "-m",
                    train_module,
                    str(seed)
                ]

            )

            if result.returncode != 0:

                print()
                print(f"{algorithm} training failed for seed {seed}")

            else:

                print()
                print(f"{algorithm} training completed for seed {seed}")

    print()
    print("=" * 60)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 60)


if __name__ == "__main__":

    main()