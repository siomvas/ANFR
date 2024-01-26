from nvflare import SimulatorRunner
import sys
RESULT_ROOT = "/path/to/sharedscratch/fednet"
algorithms_dir = "/path/to/FL_bench/fl_bench/fed_isic/job_configs/fed_isic_federated_v2"


def run_simulator():
    simulator = SimulatorRunner(
        job_folder=algorithms_dir,
        workspace=RESULT_ROOT,
        n_clients=6,
        gpu="4,4,4,5,5,5",
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    status = run_simulator()
    sys.exit(status)