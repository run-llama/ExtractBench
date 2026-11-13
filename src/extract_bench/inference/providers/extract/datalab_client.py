"""Expose submission identifiers before the Datalab SDK starts polling."""

from typing import Any

from datalab_sdk import AsyncDatalabClient

from extract_bench.inference.events import emit_provider_event


class TracingDatalabClient(AsyncDatalabClient):
    async def _submit_with_retry(self, endpoint: str, data: Any = None, json: Any = None) -> dict[str, Any]:
        response = await super()._submit_with_retry(endpoint, data=data, json=json)
        if response.get("request_check_url") is not None:
            emit_provider_event(
                "vendor.job_created",
                provider="datalab",
                stage="extract",
                job_id=response.get("request_id"),
                request_check_url=response["request_check_url"],
            )
        return response
