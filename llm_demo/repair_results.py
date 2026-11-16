"""Reproducible repair sequence; preserves original data and experiment results."""
from migrate_metrics import main as migrate
from repair_undefined_tau import main as repair_undefined
from refresh_reports import main as refresh
if __name__ == "__main__":
    migrate()
    repair_undefined()
    refresh()
