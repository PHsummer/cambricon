from setuptools import setup, find_packages
from cndb import __version__


def get_package_version():
    return __version__


setup(
    name="cndb",
    version=get_package_version(),
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[
        "sqlalchemy==1.3.23",
        "sqlalchemy_utils==0.36.8",
        "psycopg2-binary>=2.8.6",
        "pyyaml>=5.4.1",
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'cndb_submit = cndb.launch:run_commandline'
        ]
    }
)
