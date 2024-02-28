from nvflare import SimulatorRunner
import sys
RESULT_ROOT = "/path/to/sharedscratch/miccai23/colon/"
algorithms_dir = "/path/to/FL_bench/fl_bench/colon/job_configs/colon_federated"


def run_simulator():
    simulator = SimulatorRunner(
        job_folder=algorithms_dir,
        workspace=RESULT_ROOT,
        n_clients=2,
        gpu="0,0",
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    status = run_simulator()
    sys.exit(status)