def test_cli_test(script_runner):
    ret = script_runner.run("bmws", "test")
    assert ret.success


def test_cli_analyze(script_runner):
    # TODO: make a non-trivial analysis
    ret = script_runner.run("bmws", "analyze", "-h")
    assert ret.success
