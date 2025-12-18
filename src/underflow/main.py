import asyncio
import random
import re
from pathlib import Path
from typing import Callable

from rich.spinner import Spinner
from rich.style import Style
from terminaltexteffects.effects.effect_errorcorrect import ErrorCorrect
from textual import events, on, work
from textual.app import App, ComposeResult, RenderResult
from textual.containers import (
    Container,
    Horizontal,
    VerticalGroup,
    VerticalScroll,
)
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.visual import VisualType
from textual.widgets import (
    Button,
    ContentSwitcher,
    DirectoryTree,
    Input,
    Label,
    Markdown,
    ProgressBar,
    Rule,
    Static,
    TextArea,
)
from textual.widgets.text_area import TextAreaTheme
from textual.worker import Worker, get_current_worker
from textualeffects.effects import effects  # type: ignore[import-untyped]
from textualeffects.widgets import EffectLabel  # type: ignore[import-untyped]

from .model import CACHE_DIR, Model, ensure_model_ready


def _load_examples() -> list[str]:
    examples_path = Path(__file__).parent.joinpath("examples.txt")
    examples = []
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                examples.append(line)
    random.shuffle(examples)
    return examples


def _construct_text(words: list[str], intermediates: list[list[str]]):
    assert len(intermediates) < len(words)
    markup_words: list[str] = [f"[bold]{words[0]}[/bold]"]
    for intermediate_path, next_word in zip(intermediates, words[1:], strict=False):
        markup_words.extend(f"[dim]{w}[/dim]" for w in intermediate_path)
        markup_words.append(f"[bold]{next_word}[/bold]")
    return " ".join(markup_words)


class SpinnerWidget(Static):
    def __init__(self, classes: str | None = None):
        super().__init__("", classes=classes)
        self._spinner = Spinner("dots8Bit")
        self.id = "loading-spinner"

    def on_mount(self) -> None:
        self.update_render = self.set_interval(1 / 60, self.update_spinner)

    def update_spinner(self) -> None:
        self.update(self._spinner)


class GeneratingLabel(Label):
    DEFAULT_CSS = """
    GeneratingLabel {
        width: 1fr;
    }
    """

    step: reactive[int] = reactive(0)
    total_steps: reactive[int] = reactive(0)

    def render(self) -> RenderResult:
        return f"Generating words ({self.step}/{self.total_steps})"


class GeneratingWidget(Horizontal):
    step: reactive[int] = reactive(0)
    total_steps: reactive[int] = reactive(0)
    active: reactive[bool] = reactive(False)
    done: reactive[bool] = reactive(False)

    def __init__(self):
        super().__init__(id="generating")

    def compose(self) -> ComposeResult:
        yield SpinnerWidget(classes="hidden")
        yield GeneratingLabel(classes="hidden").data_bind(
            GeneratingWidget.step, GeneratingWidget.total_steps
        )
        yield Label("Done generating", id="done", classes="hidden")

    def watch_active(self, _old: bool, active: bool) -> None:
        if active:
            self.done = False
        self.query_exactly_one(SpinnerWidget).set_class(not active, "hidden")
        self.query_exactly_one(GeneratingLabel).set_class(not active, "hidden")

    def watch_done(self, _old: bool, done: bool) -> None:
        self.query_exactly_one("#done").set_class(not done, "hidden")


class ButtonsWidget(Horizontal):
    def __init__(self, *args):
        super().__init__(*args)
        self.id = "buttons-widget"


class ConfirmOverwriteModal(ModalScreen[bool]):
    def __init__(self, path: Path):
        self.path = path
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="confirm-overwrite-container", classes="modal"):
            yield Label(
                f"File '{self.path.name}' already exists.\nOverwrite?",
                id="confirm-overwrite-label",
            )
            with Horizontal(classes="modal-buttons"):
                yield Button("Yes", variant="error", id="confirm-yes")
                yield Static(classes="button-gap")
                yield Button("No", variant="primary", id="confirm-no")

    @on(Button.Pressed, "#confirm-yes")
    def on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def on_no(self) -> None:
        self.dismiss(False)


class SaveFileModal(ModalScreen[Path]):
    def compose(self) -> ComposeResult:
        with VerticalGroup(id="save-modal-container", classes="modal"):
            yield Label("Select directory and enter filename:")
            with Horizontal(classes="nav-controls"):
                yield Button("View parent", id="up-btn", classes="header-button")
            yield DirectoryTree(str(Path.cwd()), id="file-tree")
            yield Input(placeholder="Filename...", id="filename-input")
            with Horizontal(classes="modal-buttons"):
                yield Button("Save", id="save-btn", classes="header-button")
                yield Button("Cancel", id="cancel-btn", classes="header-button")

    def _get_default_filename(self, directory: Path) -> str:
        base_name = "output.txt"
        if not (directory / base_name).exists():
            return base_name

        i = 1
        while True:
            name = f"output_{i}.txt"
            if not (directory / name).exists():
                return name
            i += 1

    def on_mount(self) -> None:
        tree = self.query_exactly_one(DirectoryTree)
        self.query_exactly_one(Input).value = self._get_default_filename(
            Path(tree.path)
        )

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self.query_exactly_one(Input).value = self._get_default_filename(event.path)

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        # Show path relative to the current root of the tree
        tree = self.query_exactly_one(DirectoryTree)
        try:
            rel_path = event.path.relative_to(tree.path)
            self.query_exactly_one(Input).value = str(rel_path)
        except ValueError:
            self.query_exactly_one(Input).value = event.path.name

    @on(Button.Pressed, "#up-btn")
    def on_up_btn(self) -> None:
        tree = self.query_exactly_one(DirectoryTree)
        current = Path(tree.path)
        parent = current.parent
        if parent != current:
            tree.path = str(parent)

    @on(Button.Pressed, "#save-btn")
    def on_save(self) -> None:
        filename = self.query_exactly_one(Input).value
        if not filename:
            self.notify("Please enter a filename", severity="error")
            return

        tree = self.query_exactly_one(DirectoryTree)
        directory = Path(tree.path)

        # If user typed a relative path (e.g. subfolder/file.txt), resolve it
        full_path = directory / filename

        if full_path.exists():

            def confirm_callback(should_overwrite: bool | None) -> None:
                if should_overwrite:
                    self.dismiss(full_path)

            self.app.push_screen(ConfirmOverwriteModal(full_path), confirm_callback)
        else:
            self.dismiss(full_path)

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)


class InputTextArea(TextArea):
    check_spelling: Callable[[str], bool] = lambda _: True
    invalid_words: set[str] = set()

    async def _on_key(self, event: events.Key) -> None:
        if event.character is None or not event.character.isprintable():
            return
        if not (event.character.isalpha() or event.character == " "):
            self.notify(f"Invalid character: {event.character!r}", severity="warning")
        else:
            self.insert(event.character.lower())
        event.prevent_default()

    def _build_highlight_map(self) -> None:
        self._line_cache.clear()
        highlights = self._highlights
        highlights.clear()
        self.invalid_words.clear()

        for row, line in enumerate(self.document.lines):
            for match in re.finditer(r"[a-zA-Z]+", line):
                word = match.group(0)
                if not self.check_spelling(word.lower()):
                    start = match.start()
                    end = match.end()
                    highlights[row].append((start, end, "error"))
                    self.invalid_words.add(word)

    def remove_punctuation(self) -> None:
        self.text = re.sub(r"[^a-z\s]", "", self.text.lower())
        self._build_highlight_map()


class CompletionWidget(Static):
    _raw_renderable: object = ""

    def update(self, content: VisualType = "", *, layout: bool = True) -> None:
        self._raw_renderable = content
        super().update(content, layout=layout)

    @property
    def plain_text(self) -> str:
        from rich.text import Text

        r = self._raw_renderable
        if isinstance(r, str):
            try:
                return Text.from_markup(r).plain
            except Exception:
                return r
        if isinstance(r, Text):
            return r.plain
        return str(r)


class CLI(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("escape", "quit_app", "Quit"),
        ("ctrl+s", "save_output", "Save Output"),
    ]

    def action_quit_app(self):
        self.exit(0)

    _widgets: list[Static] = []
    _model: Model | None = None
    _examples: list[str] = _load_examples()
    _example_index: int = 0
    _current_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        with ContentSwitcher(initial="installing"):
            # installation screen
            with Container(id="installing"):
                yield Static(
                    "Installing...... (do not close the app)\n", id="installing-text"
                )
                yield Static(id="loading-explanation")
                yield ProgressBar(id="loading-bar")
                yield Static(f"[dim]Cache directory: {CACHE_DIR}", id="loading-config")

            # loading screen
            with Container(id="loading"):
                effects["ErrorCorrect"] = ErrorCorrect
                config = {"error_pairs": 1.0, "swap_delay": 30, "movement_speed": 0.5}
                label = EffectLabel(
                    "Loading......\n", effect="ErrorCorrect", config=config
                )
                label.id = "loading-text"
                yield label
                yield Static(f"[dim]Cache directory: {CACHE_DIR}", id="loading-config")

            # main screen
            with Container(id="main"):
                with Container(id="app-grid"):
                    with Container(id="header-container"):
                        yield InputTextArea(
                            id="text-input",
                            placeholder="Enter text...",
                        )
                        yield GeneratingWidget()
                        yield ButtonsWidget(
                            Button("Submit", id="submit", classes="header-button"),
                            Static(classes="button-gap"),
                            Button(
                                "Generate example",
                                id="generate-example",
                                classes="header-button",
                            ),
                            Button(
                                "Clear output",
                                id="clear-screen",
                                classes="header-button",
                            ),
                        )
                    with VerticalScroll(id="output-container"):
                        logo = (
                            Path(__file__)
                            .parent.joinpath("logo.txt")
                            .read_text(encoding="utf-8")
                        )
                        welcome_text = (
                            Path(__file__)
                            .parent.joinpath("welcome.md")
                            .read_text(encoding="utf-8")
                        )
                        yield VerticalGroup(id="results-container")
                        yield Rule(
                            line_style="dashed", id="separator", classes="hidden"
                        )
                        yield Static(logo, id="logo")
                        yield Markdown(welcome_text)

    # submit button
    @on(Button.Pressed, "#submit")
    async def on_submit_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        if button.label == "Cancel":
            if self._current_worker:
                self._current_worker.cancel()
                self._current_worker = None
            return

        text_area = self.query_exactly_one(InputTextArea)
        text_area.remove_punctuation()
        if text_area.invalid_words:
            self.notify(
                f"Invalid words: {', '.join(map(repr, text_area.invalid_words))}",
                severity="warning",
            )
            return

        button.label = "Cancel"
        self.query_exactly_one("#generate-example", Button).disabled = True
        self.query_exactly_one("#clear-screen", Button).disabled = True
        self.query_exactly_one(InputTextArea).disabled = True
        self._current_worker = await self.handle_input(text_area.text)

    # generate example button
    @on(Button.Pressed, "#generate-example")
    def on_generate_example_pressed(self) -> None:
        text_area = self.query_exactly_one(InputTextArea)
        if self._examples:
            example_text = self._examples[self._example_index]
            text_area.text = example_text
            self._example_index = (self._example_index + 1) % len(self._examples)

    # clear screen button
    @on(Button.Pressed, "#clear-screen")
    async def on_clear_screen_pressed(self) -> None:
        self._widgets.clear()
        await self.query_exactly_one("#results-container").remove_children()
        self.query_exactly_one("#separator").add_class("hidden")
        self.notify("Output cleared!")

    def action_save_output(self) -> None:
        def save_callback(path: Path | None) -> None:
            if path:
                try:
                    # Extract text from widgets in results-container
                    lines = []
                    for widget in self.query(CompletionWidget):
                        lines.append(widget.plain_text)

                    content = "\n\n".join(lines)
                    path.write_text(content, encoding="utf-8")
                    self.notify(f"Saved to {path}")
                except Exception as e:
                    self.notify(f"Error saving file: {e}", severity="error")

        self.push_screen(SaveFileModal(), save_callback)

    def on_mount(self) -> None:
        text_area = self.query_exactly_one(InputTextArea)
        theme = TextAreaTheme(
            "error_theme", syntax_styles={"error": Style(color="red", underline=True)}
        )
        text_area.register_theme(theme)
        text_area.theme = "error_theme"
        self.load_model()

    @work(exclusive=True, thread=True)
    def load_model(self) -> None:
        def update_loading_text(text: str) -> None:
            self.call_from_thread(
                self.query_exactly_one("#loading-explanation", Static).update, text
            )

        def update_progress_bar(**kwargs) -> None:
            self.call_from_thread(
                self.query_exactly_one("#loading-bar", ProgressBar).update,
                **kwargs,
            )

        ensure_model_ready(update_loading_text, update_progress_bar)
        self.query_exactly_one(ContentSwitcher).current = "loading"
        self._model = Model()
        self.query_exactly_one(InputTextArea).check_spelling = self._model.is_word
        self.query_exactly_one(ContentSwitcher).current = "main"

    @work(exclusive=True, thread=True)
    async def complete_text(self, text_input: str, widget: Static) -> None:
        if not self._model:
            return
        words = text_input.split()
        intermediates: list[list[str]] = []
        generating_widget = self.query_exactly_one(GeneratingWidget)
        generating_widget.active = True
        generating_widget.step = 0
        generating_widget.total_steps = len(words) - 1

        try:
            for i in range(len(words) - 1):
                if get_current_worker().is_cancelled:
                    break
                generating_widget.step += 1

                path = self._model.find_best_path(
                    words[i],
                    words[i + 1],
                    step_distances=[x / 10.0 for x in range(1, 20)],
                )

                base_text = _construct_text(words, intermediates)

                # Animate intermediate words (dimmed)
                for w in path[1:-1]:
                    base_text += " "
                    for k in range(1, len(w) + 1):
                        if get_current_worker().is_cancelled:
                            break
                        self.call_from_thread(
                            widget.update, base_text + f"[dim]{w[:k]}[/dim]"
                        )
                        await asyncio.sleep(0.02)
                    base_text += f"[dim]{w}[/dim]"

                # Animate target word (bold)
                target_word = path[-1]
                base_text += " "
                for k in range(1, len(target_word) + 1):
                    if get_current_worker().is_cancelled:
                        break
                    self.call_from_thread(
                        widget.update, base_text + f"[bold]{target_word[:k]}[/bold]"
                    )
                    await asyncio.sleep(0.02)

                intermediates.append(path[1:-1])
        finally:
            generating_widget.active = False
            # If we completed the loop (not cancelled), show done.
            if not get_current_worker().is_cancelled:
                generating_widget.done = True
                self.call_from_thread(
                    widget.update, _construct_text(words, intermediates)
                )

            self.query_exactly_one(InputTextArea).disabled = False  # re-enable input
            self.query_exactly_one("#submit", Button).label = "Submit"
            self.query_exactly_one("#generate-example", Button).disabled = False
            self.query_exactly_one("#clear-screen", Button).disabled = False

    async def handle_input(self, value: str) -> Worker:
        widget = CompletionWidget(classes="completion")
        self._widgets.append(widget)
        await self.query_exactly_one("#results-container").mount(widget)
        self.query_exactly_one("#separator").remove_class("hidden")
        return self.complete_text(value, widget)


def main():
    app = CLI()
    app.run()


if __name__ == "__main__":
    main()
