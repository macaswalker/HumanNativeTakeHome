from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from src.presidio.detector import presidio_detect

anonymizer = AnonymizerEngine()

def presidio_anonymize(text, entities=None, language="en"):
    analysis = presidio_detect(text, entities, language)

    operators = {
        result.entity_type: OperatorConfig("replace", {"new_value": f"[{result.entity_type}]"})
        for result in analysis
    }

    return anonymizer.anonymize(
        text=text,
        analyzer_results=analysis,
        operators=operators
    )
