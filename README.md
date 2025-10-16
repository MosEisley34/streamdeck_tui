# streamdeck_tui

Streamdeck TUI is a [Textual](https://textual.textualize.io/)-powered interface for managing IPTV providers from the terminal. It
combines provider configuration, playlist loading, and channel browsing into a responsive dashboard that keeps network work off
the UI thread.

## Features

- Maintain multiple IPTV providers with editable name, playlist, and status API settings.
- Persist provider configuration to `config.yaml` and reload changes instantly.
- Fetch playlists and connection status in background workers so the UI stays responsive.
- Browse channels with lightning-fast incremental search and rich metadata display.
- Observe provider health with active/max connection counts when a status API is configured.

## Installation

Create and activate a Python 3.11+ environment, then install the project and development extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

Use any virtual environment location you prefer—the `.venv` directory is only a suggestion. Once activated, the `Makefile`
targets shown above install the editable package along with the optional development dependencies.

## Cloning the repository

To work on the application locally, clone the repository and enter the project directory:

```bash
git clone https://github.com/<your-account>/streamdeck_tui.git
cd streamdeck_tui
```

Replace `<your-account>` with the GitHub namespace that hosts your fork if you are not cloning directly from the upstream
repository. After cloning you can use the convenience targets defined in the `Makefile`:

```bash
make install   # install editable project + dev tools
make run       # launch the TUI (equivalent to python -m streamdeck_tui)
make test      # run pytest
```

The `streamdeck-tui` console entry point is installed automatically when the package is installed. If you prefer not to install
the script, you can invoke the package directly:

```bash
python -m streamdeck_tui
```

When working directly from a fresh clone, you can also execute the Textual app module without installing the package first. Run
either of the following commands from the project root (the second assumes you have changed into `src/streamdeck_tui`):

```bash
python src/streamdeck_tui/app.py
# or
python app.py
```

Both forms ensure the package-relative imports resolve correctly before launching the UI.

Use `streamdeck-tui --help` (or the module equivalent) for runtime flags such as `--config` to point to a different configuration
file. Run `pytest` to execute the test suite.

## Configuration

Configuration lives at `~/.config/streamdeck_tui/config.yaml` by default (override with `--config`). Define one or more providers
under a top-level `providers` key:

```yaml
providers:
  - name: Example IPTV
    playlist_url: https://example.com/playlist.m3u
    api_url: https://example.com/status  # optional JSON endpoint with connection counts
```

Each provider entry requires a unique `name` and `playlist_url`. The optional `api_url` should return JSON containing
`active_connections` and `max_connections` keys to populate the provider status line.

## Usage

Launch the TUI with:

```bash
streamdeck-tui
```

> **Important:** If you have run `pip install streamdeck-tui` previously, the
> `streamdeck-tui` command may still point at that older installation. Always run
> `make install` (or `pip install -e .`) after cloning so the console entry point
> is refreshed to use the code in this workspace. You can also invoke the module
> directly with `python -m streamdeck_tui` or `make run` to bypass any lingering
> system-wide installation.

Keyboard shortcuts:

| Key | Action |
| --- | --- |
| `q` | Quit |
| `/` | Focus the search box |
| `Esc` | Clear the current channel search |
| `n` | Start configuring a new provider |
| `Ctrl+S` | Save provider changes |
| `Delete` | Remove the selected provider |
| `r` | Reload playlist and status for the selected provider |

Provider changes are written to disk immediately, and selecting a provider triggers background tasks that fetch its playlist and
optional status API without blocking the interface. Channels appear in the right pane with metadata and instantaneous filtering.

### Troubleshooting startup errors

If Textual reports a stylesheet error that references
`.../site-packages/streamdeck_tui/streamdeck.css`, you are launching an older
installation that still bundles the deprecated CSS file. Fix it by running:

```bash
pip uninstall -y streamdeck-tui
make install
```

or, if you prefer not to install the console script, execute the module directly
with `python -m streamdeck_tui`. The current release bundles a Textual-compatible
stylesheet, but older installations shipped a version that relied on unsupported
properties. After reinstalling you will receive the updated CSS automatically.

## Collecting logs

Streamdeck TUI enables detailed logging by default so you can capture diagnostics while testing playlists and provider APIs.
Runtime messages are written both to stderr and to `~/.cache/streamdeck_tui.log` unless you override the location. Tweak the
verbosity and log destination with environment variables:

```bash
export STREAMDECK_TUI_LOG_LEVEL=DEBUG        # emit verbose diagnostics (INFO by default)
export STREAMDECK_TUI_LOG_FILE=/tmp/tui.log  # choose a custom log file path
```

Set the variables before running `streamdeck-tui` (or before using `make run`) to persist the configuration for your shell
session. Attach the resulting log output when reporting issues so errors surfaced by Textual, playlist downloads, or provider
status checks can be inspected quickly.

## Development

Run the tests with:

```bash
pytest
```

The test suite covers configuration persistence, playlist parsing, filtering, and provider status utilities.
