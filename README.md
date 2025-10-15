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

If you want to keep a project-specific virtual environment under `/home/carlos/python_venvs/streamdeckcli_github`, create it and
activate it with:

```bash
mkdir -p /home/carlos/python_venvs
python3 -m venv /home/carlos/python_venvs/streamdeckcli_github
source /home/carlos/python_venvs/streamdeckcli_github/bin/activate
```

Once activated, upgrade `pip` if desired and install the project dependencies with `pip install -e .[dev]` as shown above.

## Cloning the repository

To work on the application locally, clone the repository and enter the project directory:

```bash
git clone https://github.com/<your-account>/streamdeck_tui.git
cd streamdeck_tui
```

Replace `<your-account>` with the GitHub namespace that hosts your fork if you are not cloning directly from the upstream
repository. With the repository cloned and your virtual environment activated, run `streamdeck-tui` to launch the interface or
`pytest` to execute the test suite.

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
