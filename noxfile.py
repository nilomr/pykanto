import nox


@nox.session(python=["3.8", "3.9"], venv_backend="conda")
def tests(session):
    session.install("flit")
    session.run("flit", "install")
    session.run("pytest")
