import contextlib

import uvicorn

from mkdocs_smart_docs.server import app


def main():
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    )
    with contextlib.suppress(KeyboardInterrupt, SystemExit):
        uvicorn_server.run()


if __name__ == "__main__":
    main()
