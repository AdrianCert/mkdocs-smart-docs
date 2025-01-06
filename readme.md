# mkdocs-smart-docs

**mkdocs-smart-docs** is a MkDocs plugin that brings AI-powered interactivity to your documentation. With this plugin, you can add a smart button to your documentation pages, allowing users to query an NLP model for answers based on the content of the documentation.  

## Installation

Install the plugin using `pip`:

```sh
pip install mkdocs-smart-docs
```

## Usage

1. Add mkdocs-smart-docs to your mkdocs.yml configuration:

```yml
plugins:
  - smart-docs
```

2. Configure the plugin (optional):

```yml
plugins:
  - smart-docs:
      model_endpoint: "https://your-nlp-model-endpoint"
      button_text: "Ask AI"
      button_style: "primary"
```

* `model_endpoint`: URL of the NLP model API.

* `button_text`: Text displayed on the button (default: "Ask AI").

* `button_style`: CSS class for styling the button (default: "primary").

3. Build your MkDocs site:

```sh
mkdocs serve
```

4. Serve your site locally for testing:

```sh
mkdocs serve
```

## How It Works

When users click the "Ask AI" button on your documentation page:

1. The plugin open an window where user can type a query/prompt.
2. The plugin sends the query and relevant documentation context to your specified NLP model endpoint.
3. The model processes the input and returns an accurate response.
4. The answer is displayed interactively for the user.

## Customization

The button and interaction can be styled and extended to match your documentation's design. For advanced usage, refer to the [documentation](#).

## Contributing

Contributions are welcome! If you’d like to report issues or suggest new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
