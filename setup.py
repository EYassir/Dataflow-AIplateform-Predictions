import setuptools

REQUIRED_PACKAGES = [
    'apache-beam[gcp]',
    'tensorflow-transform',
    'tensorflow',
]

setuptools.setup(
    name='molecules',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description='Cloud ML molecules sample with preprocessing',
)