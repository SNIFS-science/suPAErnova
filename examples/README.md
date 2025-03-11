# SuPAErnova

## General Workflow
`SuPAErnova` has been designed around a workflow that should allow for simple configuration when wanting to 'just run' the PAE model, whilst still enabling flexibility and modularity when you want to tinker and break things. To this end, there are many ways you can interact with `SuPAErnova`, which will be explored in the following examples.

In general, each step goes through the following stages (all of which are callback-able):

1. Setup: Each step has a `setup()` function which initialises the step, and reads in values from the user-provided configuration.
2. Run: Each step has a `run()` function which can be considered the "main" function of the step, doing all of the actual work of this step. Note that, unless the `--force` option is passed, this step will first check whether existing results from a previous run exist, and will prefer to just load those results in, rather than rerunning.
3. Result: Each step has a `result()` function which saves the result of the `run()` function to your output path, as well as recording them in the config dictionary, allowing the results to be used by later steps.

### `SuPAErnova` as a script
If you just wish to run `SuPAErnova` on some data, your best bet is to write a `.toml` file describing your intentions. You can then run `SuPAErnova` on this `.toml` file via:

```
uv run suPAErnova path/to/your/suPAErnova.toml
```

Of course, if you're not using `uv` you will need to handle your dependency management and virtual environment appropriately. The provided [pyproject.toml](../pyproject.toml) should let you set up a virtual environment using whatever your preferred tool is, allowing you to instead run `suPAErnova` via:

```
source ./venv/bin/activate # Or however you activate your virtual environment
python src/suPAErnova/__init__.py path/to/your/suPAErnova.toml
```

The examples provided here include `.toml` files containing explanations for the configuration of each `Step` in `SuPAErnova`.

### `SuPAErnova` as a library

If you wish to integrate SuPAErnova into your own projects, rather than run it as an independent script, the workflow is fairly similar. Rather than writing up a `suPAErnova.toml` config file, you just need to build up an equivalent dictionary, and then run `SuPAErnova` as normal. Each example has a Jupyter notebook detailing how to use `SuPAErnova` in this way.

### Callback functions

If you want to add additional functionality without needing to delve into the source code, each step has been split into sub-steps, each of which can have an *arbitrary* user-defined function run before, and after, the sub-step, modifying its behaviour. These callback functions should be placed in a python script, which is then included in the `suPAErnova.toml` config file. Alternatively, the function can be passed directly into the `SuPAErnova` dictionary if running it as a library. An example of using callback functions is provided for each step.

## Naming conventions

Throughout the documentation, and the source code, the following naming conventions are used:


