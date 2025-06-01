import logging
import neptune


class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing:"
            )
        )
    
neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
    _FilterCallback()
)