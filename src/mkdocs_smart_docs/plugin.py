import collections
import mimetypes
import shutil
from pathlib import Path

from mkdocs.plugins import BasePlugin

from .nlp import QuestionAnsweringSupervisor


class SmartDocsPlugin(BasePlugin):
    """
    A MkDocs plugin to enhance documentation with smart features such as
    automatic asset preparation and content scraping for question-answering models.

    See https://www.mkdocs.org/user-guide/plugins/ for more information.
    https://samrambles.com/assets/images/plugin-events.svg
    """

    def _prepare_assets(self, config):
        site_dir = Path(config.get("site_dir")).resolve()
        plugin_resources = Path(__file__, "../assets/site")
        walk_queue = collections.deque([plugin_resources])

        while walk_queue:
            current = walk_queue.popleft()
            if current.is_dir():
                walk_queue.extend(current.iterdir())
                continue

            rel_path = Path("assets", current.relative_to(plugin_resources)).as_posix()
            type, _ = mimetypes.guess_type(current)
            if type in ("text/css",):
                config.setdefault("extra_css", []).append(rel_path)
            elif type in ("application/javascript",):
                config.setdefault("extra_javascript", []).append(rel_path)
            else:
                # Skip unknown types, maybe we should add later
                continue
            dest = site_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(current, dest)

    def scrape_content(self, content):
        docs = {}
        for file in content:
            # TODO: Add parsing others file types
            if file.src_path.endswith(".md"):
                docs[file.abs_src_path] = file.content_string
        return docs

    def on_nav(self, nav, /, *, config, files):
        self._prepare_assets(config)
        if self.config.get("scrape_content", False):
            docs = self.scrape_content(files)
            qa_supervisor = QuestionAnsweringSupervisor()
            qa_model = qa_supervisor.train(*list(docs.values()), epochs=3)
            qa_model.save(Path(config["site_dir"], "assets", "smart-docs", "qa-model"))

    def on_serve(self, server, /, *, config, builder):
        return super().on_serve(server, config=config, builder=builder)
