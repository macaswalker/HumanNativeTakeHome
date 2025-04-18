from presidio_analyzer import AnalyzerEngine

DEFAULT_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "URL"]

analyzer = AnalyzerEngine()

def presidio_detect(text, entities=None, language="en"):
    """
    Run Presidio default detection on a given text.
    """
    if entities is None:
        entities = DEFAULT_ENTITIES

    return analyzer.analyze(
        text=text,
        entities=entities,
        language=language
    )
