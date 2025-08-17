## Language Pipes

[![GitHub license](License-Image)](License-Url)
[![Release][Release-Image]][Release-Url] 
![PyPI - Version](PyPiVersion-Url)
![PyPI - Python Version](PythonVersion-Url)

[License-Image]: https://img.shields.io/badge/license-MIT-blue.svg
[License-Url]: https://github.com/erinclemmer/language-pipes/blob/main/LICENSE

[Release-Url]: https://github.com/erinclemmer/language-pipes/releases/latest
[Release-Image]: https://img.shields.io/github/v/release/erinclemmer/language-pipes

[PyPiVersion-Url]: https://img.shields.io/pypi/v/language-pipes
[PythonVersion-Url]: https://img.shields.io/pypi/pyversions/language-pipes

Language pipes is an application designed to allow more people to have access to local language models. Over the past few years open source language models have become much more powerful yet the most powerful models are still out of reach of the general population because of the extreme amounts of ram that is needed to host these models. Language pipes allows multiple computer systems to host the same model and move computatiton data between them while being as easy as possible to set up.


# Installation
To install the latest version:
```bash
pip install language-pipes
```

Create a network key for the network:
```bash
language-pipes create_key network.key
```

Then create a `config.json` file to tell the program how to operate. This is an example:
```json
{
    "logging_level": "info",
    "oai_port": 8080,
    "router": {
        "node_id": "node-1",
        "port": 5000,
        "aes_key_file":"network.key",
        "bootstrap_nodes": [
            {
                "address": "192.0.0.10",
                "port": 5000
            }
        ]
    },
    "processor": {
        "https": true,
        "job_port": 5001,
        "hosted_models": [
            {
                "id": "cool-model",
                "device": "cuda",
                "max_memory": 5
            }
        ]
    }
}
```
Go [here](google.com) to see what all these configuration options do.

# Documentation

# Examples

# Contributing
