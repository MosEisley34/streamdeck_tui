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
pip install -e .[dev]
```

### Example: dedicated virtual environment

If you prefer to isolate the project in its own virtual environment, choose a directory (for example,
`~/.virtualenvs/streamdeck_tui`) and create the environment there:

```bash
python3 -m venv ~/.virtualenvs/streamdeck_tui
source ~/.virtualenvs/streamdeck_tui/bin/activate
```

Feel free to substitute a different path that fits your workflow. Once activated, upgrade `pip` if desired and install the project dependencies with `pip install -e .[dev]` as shown above.

## Cloning the repository

To work on the application locally, clone the repository and enter the project directory:

```bash
git clone https://github.com/<your-account>/streamdeck_tui.git
cd streamdeck_tui
```

Replace `<your-account>` with the GitHub namespace that hosts your fork if you are not cloning directly from the upstream
repository. With the repository cloned and your virtual environment activated, install the project (development extras optional)
and run the TUI:

```bash
pip install -e .[dev]
streamdeck-tui
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

## Development

Run the tests with:

```bash
pytest
```

The test suite covers configuration persistence, playlist parsing, filtering, and provider status utilities.
