import nox


@nox.session(python=["3.8", "3.9"], venv_backend="conda")
def tests(session):
    # same as pip install .
    session.install(".[dev]")
    # session.run('pytest')
