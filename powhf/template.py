from powhf import utils


class InitQATemplate:
    """Manages templates with fill method for init question generation."""

    def __init__(self, template):
        utils.debug_log(f"powhf.template.InitQATemplate :: Init template: {template}")
        self.template = template

    def fill(self, full_demo=""):
        """Fills in the init template."""
        utils.debug_log(
            f"powhf.template.InitQATemplate.fill :: Filling template with demo: {full_demo}"
        )
        return self.template.replace("[full_DEMO]", full_demo)


class GenerationTemplate:
    """Manages templates with fill method for prompt generation."""

    def __init__(self, template):
        utils.debug_log(
            f"powhf.template.GenerationTemplate :: Init template: {template}"
        )
        self.template = template

    def fill(self, full_demo="", input="", output=""):
        """Fills in the generation template."""
        utils.debug_log(
            f"powhf.template.GenerationTemplate.fill :: Filling with demo, input, output."
        )
        return (
            self.template.replace("[full_DEMO]", full_demo)
            .replace("[INPUT]", input)
            .replace("[OUTPUT]", output)
        )


class EvalTemplate:
    """Manages templates with fill method for evaluation."""

    def __init__(self, template):
        utils.debug_log(f"powhf.template.EvalTemplate :: Init template: {template}")
        self.template = template

    def fill(self, prompt="", full_demo="", input="", output=""):
        """Fills in the evaluation template."""
        utils.debug_log(
            f"powhf.template.EvalTemplate.fill :: Filling template with prompt, demo, input, output"
        )
        return (
            self.template.replace("[PROMPT]", prompt)
            .replace("[full_DEMO]", full_demo)
            .replace("[INPUT]", input)
            .replace("[OUTPUT]", output)
        )

    def convert_to_generation_template(self):
        """Converts the eval template to generation template."""
        utils.debug_log(
            "powhf.template.EvalTemplate.convert_to_generation_template :: Converting to generation template"
        )
        return GenerationTemplate(self.template.replace("[PROMPT]", "[APE]"))


class DemosTemplate:
    """Manages templates with fill method for demonstrations."""

    def __init__(self, template, delimiter="\n\n"):
        utils.debug_log(
            f"powhf.template.DemosTemplate :: Init template, delimiter: {delimiter}"
        )
        self.template = template
        self.delimiter = delimiter

    def fill(self, data):
        """Fills in the demonstration template."""
        utils.debug_log(
            f"powhf.template.DemosTemplate.fill :: Filling template with data, num examples: {len(data[0])}"
        )
        demos = ""
        for i, (input_, output_) in enumerate(zip(*data)):
            demos += self.template.replace("[INPUT]", input_).replace(
                "[OUTPUT]", output_
            )

            if i != len(data[0]) - 1:
                demos += self.delimiter
        utils.debug_log(f"powhf.template.DemosTemplate.fill :: Demos: {demos}")
        return demos
