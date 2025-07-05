from setuptools import setup, find_packages

setup(
    name='worldcup_predictor',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'requests',
        'psycopg2-binary',
    ],
) 