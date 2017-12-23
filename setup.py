from setuptools import setup

__version__ = "0.1.0"

install_requires = []
setup(
    # Application name:
    name="bhabana",

    # Version number (initial):
    version=__version__,

    # Application author details:
    author="Ayushman Dash",
    author_email="ayushman@neuralspace.ai",

    # Packages
    packages=["bhabana",
              "bhabana.datasets",
              "bhabana.models",
              "bhabana.utils",
              "bhabana.tools"
              "bhabana.processing"
              "bhabana.metrics"
              "bhabana.pipeline",
              "bhabana.processing"
              "bhabana.servers"
              "bhabana.trainer"],

    # Include additional files into the package
    include_package_data=True,
    install_requires=install_requires,
    # Details
    url="https://github.com/mindgarage/Ovation/wiki",

    #
    license="LICENSE.txt",
    description="Bhabana is Odia (A language mostly spoken in Odisha, "
                "a state in eastern India) for Sentiment. This is a "
                "Deep Learning based Sentiment Analysis package that extracts "
                "sentiment information form text and gives reasons for it.",

    long_description=open("README.md").read(),
    download_url="https://github.com/mindgarage/Ovation/"
             "archive/{}.tar.gz".format(__version__)
    # Dependent packages (distributions)

)

print("Welcome to Bhabana!\n")