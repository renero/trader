from distutils.core import setup

setup(name='trader',
      version='0.2.5',
      description='Trader package',
      author='J. Renero, J. Gonzalez',
      packages=['indicators', 'predictor', 'retriever', 'trader', 'updater',
                'utils', 'portfoliomgmt'],
      )
