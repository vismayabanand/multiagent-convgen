def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run the end-to-end pipeline test (requires API key or --e2e-dataset).",
    )
    parser.addoption(
        "--e2e-dataset",
        default=None,
        help="Path to a pre-generated JSONL dataset. Skips generation and runs "
             "evaluate + assertions only.  Example: --e2e-dataset ./output/run_B.jsonl",
    )
    parser.addoption(
        "--e2e-n",
        default=100,
        type=int,
        help="Number of conversations to generate when --run-e2e is active (default 100).",
    )
    parser.addoption(
        "--e2e-seed",
        default=42,
        type=int,
        help="Random seed for the generation run (default 42).",
    )
