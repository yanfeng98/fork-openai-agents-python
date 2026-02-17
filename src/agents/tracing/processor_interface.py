import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .spans import Span
    from .traces import Trace


class TracingProcessor(abc.ABC):

    @abc.abstractmethod
    def on_trace_start(self, trace: "Trace") -> None:
        """Called when a new trace begins execution.

        Args:
            trace: The trace that started. Contains workflow name and metadata.

        Notes:
            - Called synchronously on trace start
            - Should return quickly to avoid blocking execution
            - Any errors should be caught and handled internally
        """
        pass

    @abc.abstractmethod
    def on_trace_end(self, trace: "Trace") -> None:
        """Called when a trace completes execution.

        Args:
            trace: The completed trace containing all spans and results.

        Notes:
            - Called synchronously when trace finishes
            - Good time to export/process the complete trace
            - Should handle cleanup of any trace-specific resources
        """
        pass

    @abc.abstractmethod
    def on_span_start(self, span: "Span[Any]") -> None:
        """Called when a new span begins execution.

        Args:
            span: The span that started. Contains operation details and context.

        Notes:
            - Called synchronously on span start
            - Should return quickly to avoid blocking execution
            - Spans are automatically nested under current trace/span
        """
        pass

    @abc.abstractmethod
    def on_span_end(self, span: "Span[Any]") -> None:
        """Called when a span completes execution.

        Args:
            span: The completed span containing execution results.

        Notes:
            - Called synchronously when span finishes
            - Should not block or raise exceptions
            - Good time to export/process the individual span
        """
        pass

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Called when the application stops to clean up resources.

        Should perform any necessary cleanup like:
        - Flushing queued traces/spans
        - Closing connections
        - Releasing resources
        """
        pass

    @abc.abstractmethod
    def force_flush(self) -> None:
        """Forces immediate processing of any queued traces/spans.

        Notes:
            - Should process all queued items before returning
            - Useful before shutdown or when immediate processing is needed
            - May block while processing completes
        """
        pass


class TracingExporter(abc.ABC):

    @abc.abstractmethod
    def export(self, items: list["Trace | Span[Any]"]) -> None:
        """Exports a list of traces and spans.

        Args:
            items: The items to export.
        """
        pass
